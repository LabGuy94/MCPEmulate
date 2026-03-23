"""Emulation session management — Unicorn engine lifecycle and state."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from unicorn import Uc, UC_HOOK_CODE, UC_PROT_ALL, UC_PROT_READ, UC_PROT_WRITE, UC_PROT_EXEC

from .architectures import ArchConfig, get_arch

# Page size for memory alignment.
PAGE_SIZE = 0x1000

# Permission string -> UC_PROT_* bitmask mapping.
_PERM_BITS = {"r": UC_PROT_READ, "w": UC_PROT_WRITE, "x": UC_PROT_EXEC}


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

    __slots__ = ("id", "arch", "uc", "mapped_regions", "_insn_count")

    def __init__(self, session_id: str, arch: ArchConfig) -> None:
        self.id = session_id
        self.arch = arch
        self.uc = Uc(arch.uc_arch, arch.uc_mode)
        self.mapped_regions: list[MemoryRegion] = []
        self._insn_count: int = 0

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

        self._insn_count = 0

        def _code_hook(uc: Uc, address: int, size: int, user_data: object) -> None:
            self._insn_count += 1

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
            if count > 0 and self._insn_count >= count:
                stop_reason = "count_exhausted"
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
        return result


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
