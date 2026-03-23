"""Emulation session management — Unicorn engine lifecycle and state."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from unicorn import Uc, UC_HOOK_CODE, UC_HOOK_MEM_READ, UC_HOOK_MEM_WRITE, UC_MEM_WRITE, UC_MEM_READ, UC_PROT_ALL, UC_PROT_READ, UC_PROT_WRITE, UC_PROT_EXEC

from .architectures import ArchConfig, get_arch

# Page size for memory alignment.
PAGE_SIZE = 0x1000

# Permission string -> UC_PROT_* bitmask mapping.
_PERM_BITS = {"r": UC_PROT_READ, "w": UC_PROT_WRITE, "x": UC_PROT_EXEC}

# Watchpoint access type -> Unicorn hook type.
_WP_ACCESS = {
    "r": UC_HOOK_MEM_READ,
    "w": UC_HOOK_MEM_WRITE,
    "rw": UC_HOOK_MEM_READ | UC_HOOK_MEM_WRITE,
}


def parse_perms(perms: str) -> int:
    """Parse a permission string like 'rwx', 'rx', 'rw' into a UC_PROT bitmask.

    Raises ValueError on unrecognised characters or empty string.
    """
    if not perms:
        raise ValueError("Permissions string must not be empty")
    result = 0
    for ch in perms:
        bit = _PERM_BITS.get(ch)
        if bit is None:
            raise ValueError(
                f"Invalid permission character {ch!r}. Valid characters: r, w, x"
            )
        result |= bit
    return result


def align_up(value: int, alignment: int) -> int:
    """Round *value* up to the next multiple of *alignment*."""
    return (value + alignment - 1) & ~(alignment - 1)


@dataclass
class MemoryRegion:
    """A mapped memory region in the emulator."""

    address: int
    size: int
    perms: int  # UC_PROT_* bitmask


class EmulationSession:
    """Wraps a single Unicorn engine instance with bookkeeping."""

    __slots__ = ("id", "arch", "uc", "mapped_regions", "_insn_count", "breakpoints", "_contexts", "_hit_breakpoint", "_watchpoints", "_hit_watchpoint", "_trace_enabled", "_trace_log", "_trace_max", "_symbols")

    def __init__(self, session_id: str, arch: ArchConfig) -> None:
        self.id = session_id
        self.arch = arch
        self.uc = Uc(arch.uc_arch, arch.uc_mode)
        self.mapped_regions: list[MemoryRegion] = []
        self._insn_count: int = 0
        self.breakpoints: set[int] = set()
        self._contexts: dict[str, object] = {}  # label -> UcContext
        self._hit_breakpoint: int | None = None
        self._watchpoints: dict[int, tuple[int, int, str]] = {}  # addr -> (hook_handle, size, access_str)
        self._hit_watchpoint: dict | None = None
        self._trace_enabled: bool = False
        self._trace_log: list[tuple[int, bytes]] = []  # (address, insn_bytes)
        self._trace_max: int = 10_000
        self._symbols: dict[str, int] = {}  # name -> address

    # -- Memory operations ---------------------------------------------------

    def map_memory(self, address: int, size: int, perms: int = UC_PROT_ALL) -> MemoryRegion:
        """Map a memory region, rounding size up to page alignment."""
        aligned_size = align_up(size, PAGE_SIZE)
        if aligned_size == 0:
            aligned_size = PAGE_SIZE
        self.uc.mem_map(address, aligned_size, perms)
        region = MemoryRegion(address=address, size=aligned_size, perms=perms)
        self.mapped_regions.append(region)
        return region

    def write_memory(self, address: int, data: bytes) -> int:
        """Write raw bytes into the emulator's memory. Returns bytes written."""
        self.uc.mem_write(address, data)
        return len(data)

    def read_memory(self, address: int, size: int) -> bytes:
        """Read raw bytes from the emulator's memory."""
        return bytes(self.uc.mem_read(address, size))

    def list_regions(self) -> list[dict]:
        """Return all mapped memory regions."""
        return [{"address": r.address, "size": r.size, "perms": r.perms} for r in self.mapped_regions]

    def hexdump(self, address: int, size: int = 256) -> str:
        """Formatted hex dump. Size capped at 4096 bytes."""
        size = min(size, 4096)
        data = self.read_memory(address, size)
        lines = []
        for offset in range(0, len(data), 16):
            chunk = data[offset:offset + 16]
            hex_left = " ".join(f"{b:02x}" for b in chunk[:8])
            hex_right = " ".join(f"{b:02x}" for b in chunk[8:])
            ascii_repr = "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in chunk)
            addr_str = f"{address + offset:08x}"
            lines.append(f"{addr_str}  {hex_left:<23s}  {hex_right:<23s}  |{ascii_repr}|")
        return "\n".join(lines)

    def search_memory(self, pattern: bytes, address: int | None = None, size: int | None = None, max_results: int = 100) -> list[int]:
        """Search for byte pattern in memory. Returns list of match addresses."""
        matches: list[int] = []
        if address is not None and size is not None:
            # Search a specific range.
            data = self.read_memory(address, size)
            idx = 0
            while idx <= len(data) - len(pattern) and len(matches) < max_results:
                pos = data.find(pattern, idx)
                if pos == -1:
                    break
                matches.append(address + pos)
                idx = pos + 1
        else:
            # Search all mapped regions.
            for region in self.mapped_regions:
                if len(matches) >= max_results:
                    break
                data = self.read_memory(region.address, region.size)
                idx = 0
                while idx <= len(data) - len(pattern) and len(matches) < max_results:
                    pos = data.find(pattern, idx)
                    if pos == -1:
                        break
                    matches.append(region.address + pos)
                    idx = pos + 1
        return matches

    # -- Register operations -------------------------------------------------

    def _resolve_reg(self, name: str) -> int:
        """Resolve a register name to its Unicorn constant. Raises ValueError."""
        reg_id = self.arch.register_map.get(name.lower())
        if reg_id is None:
            valid = ", ".join(sorted(self.arch.register_map))
            raise ValueError(
                f"Unknown register {name!r} for {self.arch.name}. Valid: {valid}"
            )
        return reg_id

    def set_register(self, name: str, value: int) -> None:
        self.uc.reg_write(self._resolve_reg(name), value)

    def get_register(self, name: str) -> int:
        return self.uc.reg_read(self._resolve_reg(name))

    def set_registers(self, values: dict[str, int]) -> dict[str, int]:
        """Bulk-write registers. Returns the values actually written."""
        result: dict[str, int] = {}
        for name, value in values.items():
            self.set_register(name, value)
            result[name.lower()] = value
        return result

    def get_registers(self, names: list[str] | None = None) -> dict[str, int]:
        """Bulk-read registers. None means all registers."""
        if names is None:
            names = list(self.arch.register_map)
        return {name.lower(): self.get_register(name) for name in names}

    # -- Emulation -----------------------------------------------------------

    def emulate(
        self,
        address: int,
        stop_address: int = 0,
        count: int = 0,
        timeout_us: int = 0,
    ) -> dict:
        """Run emulation.

        Returns a dict with stop_reason, instructions_executed, and final register snapshot.
        At least one of *stop_address* or *count* must be nonzero.
        """
        if stop_address == 0 and count == 0:
            raise ValueError("Must provide stop_address, count, or both")

        self._hit_breakpoint = None
        self._hit_watchpoint = None
        self._insn_count = 0

        def _code_hook(uc: Uc, address: int, size: int, user_data: object) -> None:
            self._insn_count += 1
            if address in self.breakpoints:
                self._hit_breakpoint = address
                uc.emu_stop()
                return
            if self._trace_enabled and len(self._trace_log) < self._trace_max:
                insn_bytes = bytes(uc.mem_read(address, size))
                self._trace_log.append((address, insn_bytes))

        hook_handle = self.uc.hook_add(UC_HOOK_CODE, _code_hook)
        stop_reason = "completed"
        try:
            self.uc.emu_start(
                address,
                until=stop_address,
                timeout=timeout_us,
                count=count,
            )
        except Exception as exc:
            stop_reason = "error"
            # Attach error detail to the result below.
            error_detail = str(exc)
        else:
            error_detail = None
            # Determine stop reason when no exception.
            if self._hit_breakpoint is not None:
                # The hook incremented _insn_count for the breakpoint instruction
                # that never actually executed. Correct the count.
                self._insn_count -= 1
                stop_reason = "breakpoint"
            elif count > 0 and self._insn_count >= count:
                stop_reason = "count_exhausted"
            elif self._hit_watchpoint is not None:
                stop_reason = "watchpoint"
            # Unicorn signals timeout via exception in most cases, but guard:
            if timeout_us > 0 and stop_reason == "completed":
                # If we reached here with fewer instructions than count and
                # haven't hit stop_address, it might be timeout.  Unicorn 2
                # raises UcError on timeout, so this branch is mostly a safety net.
                pass
        finally:
            self.uc.hook_del(hook_handle)

        result: dict = {
            "stop_reason": stop_reason,
            "instructions_executed": self._insn_count,
            "registers": self.get_registers(),
        }
        if error_detail is not None:
            result["error_detail"] = error_detail
        if self._hit_breakpoint is not None:
            result["breakpoint_address"] = self._hit_breakpoint
        if self._hit_watchpoint is not None:
            result["watchpoint"] = self._hit_watchpoint
        return result

    # -- Breakpoint operations -----------------------------------------------

    def add_breakpoint(self, address: int) -> int:
        """Add a breakpoint. Returns total breakpoint count. Idempotent."""
        self.breakpoints.add(address)
        return len(self.breakpoints)

    def remove_breakpoint(self, address: int) -> int:
        """Remove a breakpoint. Returns total breakpoint count. Raises KeyError if not found."""
        try:
            self.breakpoints.remove(address)
        except KeyError:
            raise KeyError(f"No breakpoint at address 0x{address:x}") from None
        return len(self.breakpoints)

    def list_breakpoints(self) -> list[int]:
        """Return sorted list of breakpoint addresses."""
        return sorted(self.breakpoints)

    # -- Watchpoint operations ---------------------------------------------------

    def add_watchpoint(self, address: int, size: int = 1, access: str = "w") -> int:
        """Add a memory watchpoint. Returns total watchpoint count.

        Idempotent — same address replaces the existing watchpoint.
        """
        if access not in _WP_ACCESS:
            raise ValueError(f"Invalid access type {access!r}. Use 'r', 'w', or 'rw'.")
        # Remove existing watchpoint at this address if present.
        if address in self._watchpoints:
            self.uc.hook_del(self._watchpoints[address][0])

        def _wp_callback(uc, acc, addr, sz, value, user_data):
            self._hit_watchpoint = {
                "address": addr,
                "access": "write" if acc == UC_MEM_WRITE else "read",
                "size": sz,
                "value": value,
            }
            uc.emu_stop()

        hook_type = _WP_ACCESS[access]
        handle = self.uc.hook_add(hook_type, _wp_callback, begin=address, end=address + size - 1)
        self._watchpoints[address] = (handle, size, access)
        return len(self._watchpoints)

    def remove_watchpoint(self, address: int) -> int:
        """Remove a watchpoint. Returns total watchpoint count. Raises KeyError if not found."""
        try:
            handle, _, _ = self._watchpoints.pop(address)
        except KeyError:
            raise KeyError(f"No watchpoint at address 0x{address:x}") from None
        self.uc.hook_del(handle)
        return len(self._watchpoints)

    def list_watchpoints(self) -> list[dict]:
        """Return sorted list of watchpoints."""
        return sorted(
            [{"address": addr, "size": info[1], "access": info[2]} for addr, info in self._watchpoints.items()],
            key=lambda w: w["address"],
        )

    # -- Trace operations -------------------------------------------------------

    def enable_trace(self, max_entries: int = 10_000) -> None:
        """Enable instruction tracing, clearing any existing log."""
        self._trace_enabled = True
        self._trace_log = []
        self._trace_max = max_entries

    def disable_trace(self) -> int:
        """Disable tracing. Log is preserved. Returns entry count."""
        self._trace_enabled = False
        return len(self._trace_log)

    def get_trace(self, offset: int = 0, limit: int = 100) -> dict:
        """Get trace entries with pagination. Disassembles on the fly."""
        from capstone import Cs

        cs = Cs(self.arch.cs_arch, self.arch.cs_mode)
        total = len(self._trace_log)
        selected = self._trace_log[offset:offset + limit]
        entries = []
        for i, (addr, raw) in enumerate(selected, start=offset):
            insn = next(cs.disasm(raw, addr, count=1), None)
            entry: dict = {
                "index": i,
                "address": addr,
                "bytes_hex": raw.hex(),
                "size": len(raw),
            }
            if insn is not None:
                entry["mnemonic"] = insn.mnemonic
                entry["op_str"] = insn.op_str
            entries.append(entry)
        return {"entries": entries, "total": total, "offset": offset, "limit": limit}

    # -- Stepping ------------------------------------------------------------

    def step(self, address: int | None = None) -> dict:
        """Execute one instruction. Uses current PC if address is None.

        Returns dict with address, instruction disassembly, and registers.
        """
        from capstone import Cs

        if address is None:
            pc_reg = self.arch.pc_reg
            address = self.get_register(pc_reg)

        result = self.emulate(address=address, count=1)

        # Disassemble the single instruction that was at `address`.
        # Read enough bytes for the longest possible instruction (15 for x86).
        try:
            code = self.read_memory(address, 16)
        except Exception:
            code = self.read_memory(address, 4)  # ARM instructions are 4 bytes

        cs = Cs(self.arch.cs_arch, self.arch.cs_mode)
        insn = next(cs.disasm(code, address, count=1), None)

        step_result: dict = {
            "address": address,
            "registers": result["registers"],
            "stop_reason": result["stop_reason"],
        }
        if insn is not None:
            step_result["instruction"] = {
                "mnemonic": insn.mnemonic,
                "op_str": insn.op_str,
                "bytes_hex": insn.bytes.hex(),
                "size": insn.size,
            }
        return step_result

    # -- Context save/restore ------------------------------------------------

    def save_context(self, label: str) -> list[str]:
        """Snapshot all registers under a label. Returns all saved labels."""
        self._contexts[label] = self.uc.context_save()
        return sorted(self._contexts.keys())

    def restore_context(self, label: str) -> None:
        """Restore registers from a saved snapshot. Raises KeyError if not found."""
        try:
            ctx = self._contexts[label]
        except KeyError:
            raise KeyError(f"No saved context with label {label!r}") from None
        self.uc.context_restore(ctx)

    # -- Symbol operations ------------------------------------------------------

    def add_symbol(self, name: str, address: int) -> int:
        """Associate a name with an address. Overwrites if exists. Returns total count."""
        self._symbols[name] = address
        return len(self._symbols)

    def remove_symbol(self, name: str) -> int:
        """Remove a symbol. Returns total count. Raises KeyError if not found."""
        try:
            del self._symbols[name]
        except KeyError:
            raise KeyError(f"No symbol with name {name!r}") from None
        return len(self._symbols)

    def list_symbols(self) -> list[dict]:
        """Return list of symbols sorted by name."""
        return sorted(
            [{"name": n, "address": a} for n, a in self._symbols.items()],
            key=lambda s: s["name"],
        )

    # -- Load helper ------------------------------------------------------------

    def load_binary(self, data: bytes, address: int, entry_point: int | None = None) -> dict:
        """Load binary data: auto-map memory, write data, optionally set PC."""
        # Auto-map: align address down to page boundary, size up.
        page_addr = address & ~(PAGE_SIZE - 1)
        end = address + len(data)
        page_size = align_up(end - page_addr, PAGE_SIZE)
        if page_size == 0:
            page_size = PAGE_SIZE
        try:
            self.map_memory(page_addr, page_size)
        except Exception:
            pass  # Already mapped — write will still work
        self.write_memory(address, data)
        if entry_point is not None:
            self.set_register(self.arch.pc_reg, entry_point)
        return {
            "address": address,
            "size": len(data),
            "entry_point": entry_point,
        }


class SessionManager:
    """Manages the lifecycle of emulation sessions."""

    MAX_SESSIONS = 16

    def __init__(self) -> None:
        self._sessions: dict[str, EmulationSession] = {}

    def create(self, arch_name: str) -> EmulationSession:
        """Create a new emulation session for the given architecture name."""
        if len(self._sessions) >= self.MAX_SESSIONS:
            raise RuntimeError(
                f"Maximum number of sessions ({self.MAX_SESSIONS}) reached. "
                "Destroy an existing session first."
            )
        arch = get_arch(arch_name)
        session_id = uuid.uuid4().hex
        session = EmulationSession(session_id, arch)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> EmulationSession:
        """Retrieve a session by ID. Raises KeyError if not found."""
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(f"No session with id {session_id!r}") from None

    def destroy(self, session_id: str) -> None:
        """Destroy and remove a session. Raises KeyError if not found."""
        session = self.get(session_id)
        # Unicorn doesn't have an explicit close, but dropping the reference suffices.
        del self._sessions[session_id]
