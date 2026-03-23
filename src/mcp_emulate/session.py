"""Emulation session management — Unicorn engine lifecycle and state."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
import time
from collections import deque, defaultdict
import operator as _op

from unicorn import (
    Uc, UC_HOOK_CODE, UC_HOOK_MEM_READ, UC_HOOK_MEM_WRITE,
    UC_MEM_WRITE, UC_MEM_READ,
    UC_PROT_ALL, UC_PROT_READ, UC_PROT_WRITE, UC_PROT_EXEC,
    UC_HOOK_INTR, UC_HOOK_INSN,
)

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

# Condition expression parser: "register op immediate".
_CONDITION_PATTERN = re.compile(
    r"^\s*(\w+)\s*(==|!=|>=|<=|>|<|&)\s*(0x[0-9a-fA-F]+|\d+)\s*$"
)

_CONDITION_OPS = {
    "==": _op.eq, "!=": _op.ne, ">": _op.gt, "<": _op.lt,
    ">=": _op.ge, "<=": _op.le, "&": lambda a, b: bool(a & b),
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

    __slots__ = (
        "id", "arch", "uc", "mapped_regions", "_insn_count",
        "breakpoints", "_contexts", "_hit_breakpoint",
        "_watchpoints", "_hit_watchpoint",
        "_trace_enabled", "_trace_log", "_trace_max",
        "_symbols",
        # Feature 1: Memory snapshots
        "_memory_snapshots",
        # Feature 8: Saved traces
        "_saved_traces",
        # Feature 4: Syscall hooking
        "_syscall_hook_handle", "_syscall_log", "_syscall_mode",
        "_syscall_default_return",
    )

    def __init__(self, session_id: str, arch: ArchConfig) -> None:
        self.id = session_id
        self.arch = arch
        self.uc = Uc(arch.uc_arch, arch.uc_mode)
        self.mapped_regions: list[MemoryRegion] = []
        self._insn_count: int = 0
        # Feature 6: Conditional breakpoints — address → condition or None.
        self.breakpoints: dict[int, str | None] = {}
        self._contexts: dict[str, object] = {}  # label -> UcContext
        self._hit_breakpoint: int | None = None
        self._watchpoints: dict[int, tuple[int, int, str]] = {}  # addr -> (hook_handle, size, access_str)
        self._hit_watchpoint: dict | None = None
        self._trace_enabled: bool = False
        self._trace_log: deque[tuple[int, bytes]] = deque(maxlen=10_000)
        self._trace_max: int = 10_000
        self._symbols: dict[str, int] = {}  # name -> address
        # Feature 1: Memory snapshots
        self._memory_snapshots: dict[str, list[tuple[int, int, int, bytes]]] = {}
        # Feature 8: Saved traces
        self._saved_traces: dict[str, list[tuple[int, bytes]]] = {}
        # Feature 4: Syscall hooking
        self._syscall_hook_handle: int | None = None
        self._syscall_log: list[dict] = []
        self._syscall_mode: str | None = None
        self._syscall_default_return: int = 0

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
        """Write raw bytes into the emulator's memory. Returns bytes written.

        Raises ValueError if the target range is not within a writable region.
        """
        self._check_write_permission(address, len(data))
        self.uc.mem_write(address, data)
        return len(data)

    def _check_write_permission(self, address: int, size: int) -> None:
        """Verify every byte in [address, address+size) is covered by a writable region."""
        end = address + size
        for region in self.mapped_regions:
            r_start = region.address
            r_end = region.address + region.size
            if r_start < end and r_end > address:
                if not (region.perms & UC_PROT_WRITE):
                    raise ValueError(
                        f"Cannot write to address 0x{address:x}: region "
                        f"0x{r_start:x}-0x{r_end:x} is not writable"
                    )

    def read_memory(self, address: int, size: int) -> bytes:
        """Read raw bytes from the emulator's memory."""
        return bytes(self.uc.mem_read(address, size))

    def list_regions(self) -> list[dict]:
        """Return all mapped memory regions."""
        return [{"address": r.address, "size": r.size, "perms": r.perms} for r in self.mapped_regions]

    def hexdump(self, address: int, size: int = 256) -> str:
        """Formatted hex dump. Size capped at 4096 bytes.

        Consecutive identical lines are collapsed with '*' (like hexdump -C).
        """
        size = min(size, 4096)
        data = self.read_memory(address, size)
        raw_lines: list[tuple[str, bytes]] = []
        for offset in range(0, len(data), 16):
            chunk = data[offset:offset + 16]
            hex_left = " ".join(f"{b:02x}" for b in chunk[:8])
            hex_right = " ".join(f"{b:02x}" for b in chunk[8:])
            ascii_repr = "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in chunk)
            addr_str = f"{address + offset:08x}"
            line = f"{addr_str}  {hex_left:<23s}  {hex_right:<23s}  |{ascii_repr}|"
            raw_lines.append((line, chunk))

        output: list[str] = []
        prev_chunk: bytes | None = None
        suppressing = False
        for line, chunk in raw_lines:
            if chunk == prev_chunk:
                if not suppressing:
                    output.append("*")
                    suppressing = True
            else:
                output.append(line)
                suppressing = False
            prev_chunk = chunk
        return "\n".join(output)

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

    # -- Memory snapshots (Feature 1) ----------------------------------------

    def snapshot_memory(self, label: str) -> dict:
        """Capture full memory content of all mapped regions under a label. Overwrites if exists."""
        snap: list[tuple[int, int, int, bytes]] = []
        total_bytes = 0
        for region in self.mapped_regions:
            content = self.read_memory(region.address, region.size)
            snap.append((region.address, region.size, region.perms, content))
            total_bytes += region.size
        self._memory_snapshots[label] = snap
        return {
            "label": label,
            "regions": len(snap),
            "total_bytes": total_bytes,
            "labels": sorted(self._memory_snapshots.keys()),
        }

    def diff_memory(self, label_a: str, label_b: str) -> dict:
        """Compare two memory snapshots. Returns list of changed byte ranges."""
        snap_a = self._memory_snapshots.get(label_a)
        snap_b = self._memory_snapshots.get(label_b)
        if snap_a is None:
            raise KeyError(f"No memory snapshot with label {label_a!r}")
        if snap_b is None:
            raise KeyError(f"No memory snapshot with label {label_b!r}")

        # Build address → content dicts.
        map_a = {addr: content for addr, _, _, content in snap_a}
        map_b = {addr: content for addr, _, _, content in snap_b}

        changes: list[dict] = []
        new_regions: list[dict] = []
        removed_regions: list[dict] = []
        all_addrs = set(map_a) | set(map_b)
        for addr in sorted(all_addrs):
            a_data = map_a.get(addr)
            b_data = map_b.get(addr)
            if a_data is None:
                new_regions.append({"address": addr, "size": len(b_data)})
                continue
            if b_data is None:
                removed_regions.append({"address": addr, "size": len(a_data)})
                continue
            # Both exist — find changed byte ranges.
            min_len = min(len(a_data), len(b_data))
            i = 0
            while i < min_len:
                if a_data[i] != b_data[i]:
                    start = i
                    while i < min_len and a_data[i] != b_data[i]:
                        i += 1
                    changes.append({
                        "address": addr + start,
                        "size": i - start,
                        "old_hex": a_data[start:i].hex(),
                        "new_hex": b_data[start:i].hex(),
                    })
                else:
                    i += 1
        return {
            "label_a": label_a,
            "label_b": label_b,
            "changes": changes,
            "change_count": len(changes),
            "new_regions": new_regions,
            "removed_regions": removed_regions,
        }

    # -- Stack view (Feature 2) ----------------------------------------------

    def get_stack(self, count: int = 16) -> dict:
        """Read stack entries from SP, resolve values against symbols."""
        sp = self.get_register(self.arch.sp_reg)
        ptr_size = 8 if "64" in self.arch.name else 4
        # Determine byte order from architecture mode.
        byteorder = "big" if "be" in self.arch.name else "little"
        addr_to_symbol = {a: n for n, a in self._symbols.items()}
        entries: list[dict] = []
        for i in range(count):
            addr = sp + i * ptr_size
            try:
                raw = self.read_memory(addr, ptr_size)
            except Exception:
                break  # hit unmapped memory
            value = int.from_bytes(raw, byteorder)
            entry: dict = {"offset": i * ptr_size, "address": addr, "value": value}
            symbol = addr_to_symbol.get(value)
            if symbol is not None:
                entry["symbol"] = symbol
            entries.append(entry)
        return {"sp": sp, "pointer_size": ptr_size, "entries": entries, "count": len(entries)}

    # -- Memory map visualization (Feature 5) --------------------------------

    def memory_map(self) -> str:
        """Produce a /proc/self/maps-style memory layout with gaps and symbols."""
        # Build symbol lookup: addr → [names].
        addr_syms: dict[int, list[str]] = defaultdict(list)
        for name, addr in self._symbols.items():
            addr_syms[addr].append(name)

        def perm_str(p: int) -> str:
            return (
                ("r" if p & UC_PROT_READ else "-") +
                ("w" if p & UC_PROT_WRITE else "-") +
                ("x" if p & UC_PROT_EXEC else "-")
            )

        regions = sorted(self.mapped_regions, key=lambda r: r.address)
        lines: list[str] = []
        prev_end = 0
        for region in regions:
            if region.address > prev_end and prev_end > 0:
                gap = region.address - prev_end
                lines.append(f"  ... gap: {gap:#x} bytes ({prev_end:#010x}-{region.address:#010x}) ...")
            end = region.address + region.size
            lines.append(f"{region.address:#010x}-{end:#010x}  {perm_str(region.perms)}  {region.size:#x}")
            # Annotate symbols within this region.
            for sym_addr in sorted(addr_syms):
                if region.address <= sym_addr < end:
                    for name in addr_syms[sym_addr]:
                        lines.append(f"    {sym_addr:#010x}  {name}")
            prev_end = end
        return "\n".join(lines)

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

        t0 = time.perf_counter()
        self._hit_breakpoint = None
        self._hit_watchpoint = None
        self._insn_count = 0

        def _code_hook(uc: Uc, address: int, size: int, user_data: object) -> None:
            self._insn_count += 1
            if address in self.breakpoints:
                condition = self.breakpoints[address]
                if condition is None or self._eval_bp_condition(condition):
                    self._hit_breakpoint = address
                    uc.emu_stop()
                    return
            if self._trace_enabled:
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
            if timeout_us > 0 and stop_reason == "completed":
                elapsed_us = (time.perf_counter() - t0) * 1_000_000
                if elapsed_us >= timeout_us * 0.9:
                    stop_reason = "timeout"
        finally:
            self.uc.hook_del(hook_handle)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        result: dict = {
            "stop_reason": stop_reason,
            "instructions_executed": self._insn_count,
            "elapsed_ms": elapsed_ms,
            "registers": self.get_registers(),
        }
        if error_detail is not None:
            result["error_detail"] = error_detail
        if self._hit_breakpoint is not None:
            result["breakpoint_address"] = self._hit_breakpoint
        if self._hit_watchpoint is not None:
            result["watchpoint"] = self._hit_watchpoint
        return result

    # -- Breakpoint operations (Feature 6: conditional) ----------------------

    def add_breakpoint(self, address: int, condition: str | None = None) -> int:
        """Add a breakpoint. Optionally with a condition expression.

        Condition format: 'eax == 42', 'rax > 0x1000 and rcx != 0'.
        Supported operators: ==, !=, >, <, >=, <=, &.
        Idempotent — same address overwrites existing condition.
        """
        if condition is not None:
            self._validate_condition(condition)
        self.breakpoints[address] = condition
        return len(self.breakpoints)

    def remove_breakpoint(self, address: int) -> int:
        """Remove a breakpoint. Returns total breakpoint count. Raises KeyError if not found."""
        try:
            del self.breakpoints[address]
        except KeyError:
            raise KeyError(f"No breakpoint at address 0x{address:x}") from None
        return len(self.breakpoints)

    def list_breakpoints(self) -> list[dict]:
        """Return sorted list of breakpoints with conditions."""
        return sorted(
            [{"address": addr, "condition": cond} for addr, cond in self.breakpoints.items()],
            key=lambda b: b["address"],
        )

    def _validate_condition(self, condition: str) -> None:
        """Parse-check a condition string. Raises ValueError if invalid."""
        for clause in re.split(r"\s+and\s+|\s+or\s+", condition):
            m = _CONDITION_PATTERN.match(clause.strip())
            if not m:
                raise ValueError(
                    f"Invalid condition clause: {clause.strip()!r}. "
                    f"Expected: 'register op value' (e.g., 'eax == 42')"
                )
            self._resolve_reg(m.group(1))  # validate register name

    def _eval_bp_condition(self, condition: str) -> bool:
        """Evaluate a condition against current registers.

        Supports 'and'/'or' connectives. 'and' binds tighter than 'or'.
        """
        # Split on 'or' first (lower precedence), then 'and'.
        or_groups = re.split(r"\s+or\s+", condition)
        for or_group in or_groups:
            and_clauses = re.split(r"\s+and\s+", or_group)
            all_true = True
            for clause in and_clauses:
                m = _CONDITION_PATTERN.match(clause.strip())
                reg_val = self.get_register(m.group(1))
                op_fn = _CONDITION_OPS[m.group(2)]
                imm_val = int(m.group(3), 0)  # auto-detect hex/dec
                if not op_fn(reg_val, imm_val):
                    all_true = False
                    break
            if all_true:
                return True
        return False

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
            if acc == UC_MEM_WRITE:
                actual_value = value
            else:
                # Unicorn passes value=0 for reads; fetch actual memory content.
                raw = bytes(uc.mem_read(addr, sz))
                actual_value = int.from_bytes(raw, byteorder="little")
            self._hit_watchpoint = {
                "address": addr,
                "access": "write" if acc == UC_MEM_WRITE else "read",
                "size": sz,
                "value": actual_value,
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
        self._trace_log = deque(maxlen=max_entries)
        self._trace_max = max_entries

    def disable_trace(self) -> int:
        """Disable tracing. Log is preserved. Returns entry count."""
        self._trace_enabled = False
        return len(self._trace_log)

    def get_trace(self, offset: int = 0, limit: int = 100) -> dict:
        """Get trace entries with pagination. Disassembles on the fly."""
        from capstone import Cs

        cs = Cs(self.arch.cs_arch, self.arch.cs_mode)
        addr_to_symbol = {addr: name for name, addr in self._symbols.items()}
        total = len(self._trace_log)
        log_list = list(self._trace_log)
        selected = log_list[offset:offset + limit]
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
            symbol = addr_to_symbol.get(addr)
            if symbol is not None:
                entry["symbol"] = symbol
            entries.append(entry)
        return {"entries": entries, "total": total, "offset": offset, "limit": limit}

    # -- Trace diff (Feature 8) -----------------------------------------------

    def save_trace(self, label: str) -> dict:
        """Save current trace log under a label. Overwrites if exists."""
        self._saved_traces[label] = list(self._trace_log)
        return {
            "label": label,
            "entries": len(self._saved_traces[label]),
            "labels": sorted(self._saved_traces.keys()),
        }

    def diff_trace(self, label_a: str, label_b: str) -> dict:
        """Compare two saved traces instruction-by-instruction."""
        trace_a = self._saved_traces.get(label_a)
        trace_b = self._saved_traces.get(label_b)
        if trace_a is None:
            raise KeyError(f"No saved trace with label {label_a!r}")
        if trace_b is None:
            raise KeyError(f"No saved trace with label {label_b!r}")

        from capstone import Cs
        cs = Cs(self.arch.cs_arch, self.arch.cs_mode)

        def _disasm(addr: int, raw: bytes) -> dict:
            insn = next(cs.disasm(raw, addr, count=1), None)
            if insn:
                return {"address": addr, "mnemonic": insn.mnemonic, "op_str": insn.op_str, "bytes_hex": raw.hex()}
            return {"address": addr, "bytes_hex": raw.hex()}

        # Find common prefix length.
        common_prefix = 0
        for (a_addr, a_raw), (b_addr, b_raw) in zip(trace_a, trace_b):
            if a_addr == b_addr and a_raw == b_raw:
                common_prefix += 1
            else:
                break

        # Collect divergences (up to 50).
        max_len = max(len(trace_a), len(trace_b))
        divergences: list[dict] = []
        for i in range(common_prefix, min(max_len, common_prefix + 50)):
            a_entry = _disasm(*trace_a[i]) if i < len(trace_a) else None
            b_entry = _disasm(*trace_b[i]) if i < len(trace_b) else None
            if a_entry != b_entry:
                divergences.append({"index": i, "a": a_entry, "b": b_entry})

        return {
            "label_a": label_a,
            "label_b": label_b,
            "trace_a_length": len(trace_a),
            "trace_b_length": len(trace_b),
            "common_prefix": common_prefix,
            "divergence_index": common_prefix if common_prefix < max_len else None,
            "divergences": divergences,
        }

    # -- Syscall hooking (Feature 4) -----------------------------------------

    def hook_syscall(self, mode: str = "skip", default_return: int = 0) -> dict:
        """Install a syscall hook. Modes: 'skip' (log + return default), 'stop' (stop emulation).

        Idempotent — replaces existing hook.
        """
        from .architectures import SYSCALL_CONVENTIONS

        if mode not in ("skip", "stop"):
            raise ValueError(f"Invalid syscall mode {mode!r}. Use 'skip' or 'stop'.")

        conv = SYSCALL_CONVENTIONS.get(self.arch.name)
        if conv is None:
            raise ValueError(f"No syscall convention defined for {self.arch.name}")

        # Remove existing hook if present.
        if self._syscall_hook_handle is not None:
            self.uc.hook_del(self._syscall_hook_handle)
            self._syscall_hook_handle = None

        self._syscall_mode = mode
        self._syscall_default_return = default_return
        self._syscall_log = []

        def _syscall_intr_callback(uc, intno, user_data):
            # Filter by interrupt number if needed.
            if conv.intno_filter is not None and intno != conv.intno_filter:
                return  # not a syscall interrupt
            nr = self.get_register(conv.nr_reg)
            args = [self.get_register(r) for r in conv.arg_regs]
            self._syscall_log.append({
                "syscall_nr": nr, "args": args,
                "pc": self.get_register(self.arch.pc_reg),
            })
            if mode == "skip":
                self.set_register(conv.ret_reg, default_return)
            elif mode == "stop":
                uc.emu_stop()

        def _syscall_insn_callback(uc, user_data):
            # Used for x86_64 UC_HOOK_INSN with UC_X86_INS_SYSCALL.
            nr = self.get_register(conv.nr_reg)
            args = [self.get_register(r) for r in conv.arg_regs]
            self._syscall_log.append({
                "syscall_nr": nr, "args": args,
                "pc": self.get_register(self.arch.pc_reg),
            })
            if mode == "skip":
                self.set_register(conv.ret_reg, default_return)
            elif mode == "stop":
                uc.emu_stop()

        if conv.hook_type == UC_HOOK_INTR:
            self._syscall_hook_handle = self.uc.hook_add(UC_HOOK_INTR, _syscall_intr_callback)
        else:
            # UC_HOOK_INSN with specific instruction (x86_64 syscall).
            # Pass aux1 positionally — UcIntel.hook_add doesn't accept it as kwarg.
            self._syscall_hook_handle = self.uc.hook_add(
                UC_HOOK_INSN, _syscall_insn_callback, None, 1, 0, conv.hook_arg
            )

        return {"mode": mode, "default_return": default_return, "arch": self.arch.name}

    def unhook_syscall(self) -> dict:
        """Remove the syscall hook."""
        if self._syscall_hook_handle is not None:
            self.uc.hook_del(self._syscall_hook_handle)
            self._syscall_hook_handle = None
            self._syscall_mode = None
        return {"unhooked": True, "log_entries": len(self._syscall_log)}

    def get_syscall_log(self, offset: int = 0, limit: int = 100) -> dict:
        """Get recorded syscall invocations with pagination."""
        total = len(self._syscall_log)
        selected = self._syscall_log[offset:offset + limit]
        return {"entries": selected, "total": total, "offset": offset, "limit": limit}

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
        addr_to_symbol = {a: n for n, a in self._symbols.items()}
        symbol = addr_to_symbol.get(address)
        if symbol is not None:
            step_result["symbol"] = symbol
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

    # -- Executable loader (Feature 3) ----------------------------------------

    def load_executable(self, data: bytes, base_address: int = 0) -> dict:
        """Parse and load an executable binary (ELF, PE, or Mach-O) into the emulator.

        Auto-detects format via lief. Maps segments/sections with correct permissions,
        writes content, sets PC to entry point, and registers symbols.
        """
        import lief

        binary = lief.parse(list(data))
        if binary is None:
            raise ValueError("Failed to parse binary (unsupported or corrupt format)")

        loaded_segments: list[dict] = []
        fmt = "unknown"

        if isinstance(binary, lief.ELF.Binary):
            fmt = "elf"
            for segment in binary.segments:
                if segment.type != lief.ELF.Segment.TYPE.LOAD:
                    continue
                if segment.virtual_size == 0:
                    continue
                vaddr = segment.virtual_address + base_address
                page_addr = vaddr & ~(PAGE_SIZE - 1)
                seg_end = vaddr + segment.virtual_size
                map_size = align_up(seg_end - page_addr, PAGE_SIZE)
                if map_size == 0:
                    map_size = PAGE_SIZE
                perms = 0
                flags = segment.flags
                if lief.ELF.Segment.FLAGS.R in flags:
                    perms |= UC_PROT_READ
                if lief.ELF.Segment.FLAGS.W in flags:
                    perms |= UC_PROT_WRITE
                if lief.ELF.Segment.FLAGS.X in flags:
                    perms |= UC_PROT_EXEC
                if perms == 0:
                    perms = UC_PROT_READ
                try:
                    self.map_memory(page_addr, map_size, perms)
                except Exception:
                    pass  # overlapping or already mapped
                content = bytes(segment.content)
                if content:
                    self.uc.mem_write(vaddr, content)  # bypass perm check
                loaded_segments.append({
                    "address": vaddr, "size": segment.virtual_size,
                    "file_size": len(content), "perms": perms,
                })

        elif isinstance(binary, lief.PE.Binary):
            fmt = "pe"
            image_base = base_address if base_address else binary.optional_header.imagebase
            for section in binary.sections:
                vaddr = image_base + section.virtual_address
                page_addr = vaddr & ~(PAGE_SIZE - 1)
                vsize = section.virtual_size or len(section.content)
                seg_end = vaddr + vsize
                map_size = align_up(seg_end - page_addr, PAGE_SIZE)
                if map_size == 0:
                    map_size = PAGE_SIZE
                # PE section characteristics → permissions.
                chars = section.characteristics
                perms = UC_PROT_READ  # always readable
                if chars & 0x80000000:  # IMAGE_SCN_MEM_WRITE
                    perms |= UC_PROT_WRITE
                if chars & 0x20000000:  # IMAGE_SCN_MEM_EXECUTE
                    perms |= UC_PROT_EXEC
                try:
                    self.map_memory(page_addr, map_size, perms)
                except Exception:
                    pass
                content = bytes(section.content)
                if content:
                    self.uc.mem_write(vaddr, content)
                loaded_segments.append({
                    "address": vaddr, "size": vsize,
                    "file_size": len(content), "perms": perms,
                })

        elif isinstance(binary, lief.MachO.Binary):
            fmt = "macho"
            for segment in binary.segments:
                if segment.virtual_size == 0:
                    continue
                vaddr = segment.virtual_address + base_address
                page_addr = vaddr & ~(PAGE_SIZE - 1)
                seg_end = vaddr + segment.virtual_size
                map_size = align_up(seg_end - page_addr, PAGE_SIZE)
                if map_size == 0:
                    map_size = PAGE_SIZE
                # Mach-O: init_protection bits (1=R, 2=W, 4=X).
                ip = segment.init_protection
                perms = 0
                if ip & 1:
                    perms |= UC_PROT_READ
                if ip & 2:
                    perms |= UC_PROT_WRITE
                if ip & 4:
                    perms |= UC_PROT_EXEC
                if perms == 0:
                    perms = UC_PROT_READ
                try:
                    self.map_memory(page_addr, map_size, perms)
                except Exception:
                    pass
                content = bytes(segment.content)
                if content:
                    self.uc.mem_write(vaddr, content)
                loaded_segments.append({
                    "address": vaddr, "size": segment.virtual_size,
                    "file_size": len(content), "perms": perms,
                })

        # Set entry point.
        entry = binary.entrypoint + base_address
        self.set_register(self.arch.pc_reg, entry)

        # Register symbols.
        sym_count = 0
        if hasattr(binary, "symbols"):
            for sym in binary.symbols:
                if hasattr(sym, "name") and sym.name and hasattr(sym, "value") and sym.value > 0:
                    self.add_symbol(sym.name, sym.value + base_address)
                    sym_count += 1

        return {
            "format": fmt,
            "entry_point": entry,
            "segments_loaded": len(loaded_segments),
            "segments": loaded_segments,
            "symbols_registered": sym_count,
            "base_address": base_address,
        }

    # -- Session serialization (Feature 9) ------------------------------------

    def export_state(self) -> dict:
        """Export full session state to a JSON-serializable dict."""
        regions = []
        for region in self.mapped_regions:
            content = self.read_memory(region.address, region.size)
            regions.append({
                "address": region.address,
                "size": region.size,
                "perms": region.perms,
                "content_hex": content.hex(),
            })
        bps = [{"address": addr, "condition": cond} for addr, cond in self.breakpoints.items()]
        wps = [{"address": addr, "size": sz, "access": acc}
               for addr, (_, sz, acc) in self._watchpoints.items()]
        syms = [{"name": n, "address": a} for n, a in self._symbols.items()]
        return {
            "version": 1,
            "arch": self.arch.name,
            "regions": regions,
            "registers": self.get_registers(),
            "breakpoints": bps,
            "watchpoints": wps,
            "symbols": syms,
        }

    def import_state(self, state: dict) -> None:
        """Restore session state from an exported dict. Session must match architecture."""
        if state.get("arch") != self.arch.name:
            raise ValueError(
                f"State arch {state.get('arch')!r} does not match session arch {self.arch.name!r}"
            )
        # Restore regions and content.
        for region in state.get("regions", []):
            try:
                self.map_memory(region["address"], region["size"], region["perms"])
            except Exception:
                pass  # already mapped
            self.uc.mem_write(region["address"], bytes.fromhex(region["content_hex"]))
        # Restore registers.
        for name, value in state.get("registers", {}).items():
            try:
                self.set_register(name, value)
            except ValueError:
                pass  # skip unknown registers (forward compat)
        # Restore breakpoints.
        for bp in state.get("breakpoints", []):
            self.add_breakpoint(bp["address"], condition=bp.get("condition"))
        # Restore watchpoints.
        for wp in state.get("watchpoints", []):
            self.add_watchpoint(wp["address"], size=wp["size"], access=wp["access"])
        # Restore symbols.
        for sym in state.get("symbols", []):
            self.add_symbol(sym["name"], sym["address"])


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
