"""MCP server exposing CPU emulation, assembly, and disassembly tools."""

from __future__ import annotations

import base64
from typing import Any

from capstone import Cs
from keystone import Ks, KsError

from mcp.server.fastmcp import FastMCP

from .architectures import get_arch
from .session import SessionManager, parse_perms

mcp = FastMCP("mcp-emulate")
sessions = SessionManager()

# -- Helpers -----------------------------------------------------------------

_MAX_EMULATE_COUNT = 100_000
_MAX_TIMEOUT_MS = 60_000


def _decode_data(data: str, encoding: str) -> bytes:
    """Decode a data string from hex or base64 into bytes."""
    encoding = encoding.lower()
    if encoding == "hex":
        try:
            return bytes.fromhex(data)
        except ValueError as exc:
            raise ValueError(f"Invalid hex string: {exc}") from None
    elif encoding == "base64":
        try:
            return base64.b64decode(data, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid base64 string: {exc}") from None
    else:
        raise ValueError(f"Unsupported encoding {encoding!r}. Use 'hex' or 'base64'.")


def _encode_data(data: bytes, encoding: str) -> str:
    """Encode bytes into hex or base64 string."""
    encoding = encoding.lower()
    if encoding == "hex":
        return data.hex()
    elif encoding == "base64":
        return base64.b64encode(data).decode("ascii")
    else:
        raise ValueError(f"Unsupported encoding {encoding!r}. Use 'hex' or 'base64'.")


def _error(message: str, detail: str | None = None) -> dict[str, Any]:
    """Build a structured error response visible to the LLM."""
    result: dict[str, Any] = {"error": message}
    if detail is not None:
        result["detail"] = detail
    return result


def _exc_message(exc: Exception) -> str:
    """Extract a clean error message from an exception.

    KeyError.__str__ wraps the message in repr quotes; unwrap it.
    """
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def _perms_str(bitmask: int) -> str:
    """Convert a UC_PROT bitmask back to a human-readable string."""
    from unicorn import UC_PROT_READ, UC_PROT_WRITE, UC_PROT_EXEC

    parts = []
    if bitmask & UC_PROT_READ:
        parts.append("r")
    if bitmask & UC_PROT_WRITE:
        parts.append("w")
    if bitmask & UC_PROT_EXEC:
        parts.append("x")
    return "".join(parts) or "none"


# -- Session tools -----------------------------------------------------------


@mcp.tool()
def create_emulator(arch: str) -> dict:
    """Create a new CPU emulation session.

    Args:
        arch: Architecture name. One of: x86_32, x86_64, arm, arm64, mips32, mips32be, riscv32, riscv64.

    Returns a dict with session_id and arch.
    """
    try:
        session = sessions.create(arch)
        return {"session_id": session.id, "arch": session.arch.name}
    except (ValueError, RuntimeError) as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def destroy_emulator(session_id: str) -> dict:
    """Destroy an emulation session and free its resources.

    Args:
        session_id: The session ID returned by create_emulator.
    """
    try:
        sessions.destroy(session_id)
        return {"success": True}
    except KeyError as exc:
        return _error(_exc_message(exc))


# -- Memory tools ------------------------------------------------------------


@mcp.tool()
def map_memory(
    session_id: str, address: int, size: int, perms: str = "rwx"
) -> dict:
    """Map a memory region in the emulator.

    Size is rounded up to 4KB page alignment.

    Args:
        session_id: The session ID.
        address: Start address (must be page-aligned, i.e. multiple of 0x1000).
        size: Region size in bytes.
        perms: Permission string combining 'r', 'w', 'x'. Default "rwx".
    """
    try:
        session = sessions.get(session_id)
        perm_bits = parse_perms(perms)
        region = session.map_memory(address, size, perm_bits)
        result = {
            "address": region.address,
            "size": region.size,
            "perms": _perms_str(region.perms),
        }
        if region.size != size:
            result["rounded_up_from"] = size
        return result
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def write_memory(
    session_id: str, address: int, data: str, encoding: str = "hex"
) -> dict:
    """Write data to emulator memory.

    Args:
        session_id: The session ID.
        address: Destination address.
        data: Data as hex string (e.g. "90c3") or base64.
        encoding: "hex" (default) or "base64".
    """
    try:
        session = sessions.get(session_id)
        raw = _decode_data(data, encoding)
        written = session.write_memory(address, raw)
        return {"address": address, "bytes_written": written}
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def read_memory(
    session_id: str, address: int, size: int, encoding: str = "hex"
) -> dict:
    """Read data from emulator memory.

    Args:
        session_id: The session ID.
        address: Source address.
        size: Number of bytes to read.
        encoding: "hex" (default) or "base64".
    """
    try:
        session = sessions.get(session_id)
        raw = session.read_memory(address, size)
        return {
            "address": address,
            "size": size,
            "data": _encode_data(raw, encoding),
            "encoding": encoding,
        }
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Memory inspection tools -------------------------------------------------


@mcp.tool()
def list_regions(session_id: str) -> dict:
    """List all mapped memory regions.

    Args:
        session_id: The session ID.
    """
    try:
        session = sessions.get(session_id)
        regions = session.list_regions()
        return {
            "regions": [{"address": r["address"], "size": r["size"], "perms": _perms_str(r["perms"])} for r in regions],
            "count": len(regions),
        }
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def hexdump(session_id: str, address: int, size: int = 256) -> dict:
    """Formatted hex dump of memory.

    Standard format: ADDR | 16 hex bytes (8+8) | ASCII.  Max 4096 bytes.

    Args:
        session_id: The session ID.
        address: Start address.
        size: Number of bytes to dump (default 256, max 4096).
    """
    try:
        session = sessions.get(session_id)
        dump = session.hexdump(address, size)
        actual_size = min(size, 4096)
        result = {"hexdump": dump, "address": address, "size": actual_size}
        if size > 4096:
            result["clamped"] = True
            result["requested_size"] = size
        return result
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def search_memory(
    session_id: str, pattern: str, address: int | None = None,
    size: int | None = None, max_results: int = 100,
) -> dict:
    """Search for a byte pattern in memory.

    If address is None, searches all mapped regions.

    Args:
        session_id: The session ID.
        pattern: Hex string of bytes to search for (e.g. "deadbeef").
        address: Optional start address to limit search.
        size: Optional size of search range (required if address is set).
        max_results: Maximum matches to return (default 100).
    """
    try:
        session = sessions.get(session_id)
        raw_pattern = _decode_data(pattern, "hex")
        matches = session.search_memory(raw_pattern, address=address, size=size, max_results=max_results)
        return {"matches": matches, "count": len(matches), "truncated": len(matches) >= max_results}
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Memory snapshot tools (Feature 1) ----------------------------------------


@mcp.tool()
def snapshot_memory(session_id: str, label: str) -> dict:
    """Save a snapshot of all mapped memory under a label.

    Overwrites if the label already exists.

    Args:
        session_id: The session ID.
        label: A name for this memory snapshot.
    """
    try:
        session = sessions.get(session_id)
        return session.snapshot_memory(label)
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def diff_memory(session_id: str, label_a: str, label_b: str) -> dict:
    """Compare two memory snapshots and return changed byte ranges.

    Args:
        session_id: The session ID.
        label_a: First snapshot label.
        label_b: Second snapshot label.
    """
    try:
        session = sessions.get(session_id)
        return session.diff_memory(label_a, label_b)
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Stack view tool (Feature 2) ---------------------------------------------


@mcp.tool()
def get_stack(session_id: str, count: int = 16) -> dict:
    """Read stack entries from the current stack pointer.

    Resolves values against registered symbols.

    Args:
        session_id: The session ID.
        count: Number of stack entries to read (default 16, max 256).
    """
    try:
        session = sessions.get(session_id)
        count = min(count, 256)
        return session.get_stack(count=count)
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Memory map visualization tool (Feature 5) ------------------------------


@mcp.tool()
def memory_map(session_id: str) -> dict:
    """Produce a /proc/self/maps-style layout of the address space.

    Shows regions with permissions, gaps, and symbol annotations.

    Args:
        session_id: The session ID.
    """
    try:
        session = sessions.get(session_id)
        text = session.memory_map()
        return {"map": text, "region_count": len(session.mapped_regions)}
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Watchpoint tools --------------------------------------------------------


@mcp.tool()
def add_watchpoint(
    session_id: str, address: int, size: int = 1, access: str = "w",
) -> dict:
    """Add a memory watchpoint.

    Idempotent -- same address replaces the existing watchpoint.

    Args:
        session_id: The session ID.
        address: Memory address to watch.
        size: Number of bytes to watch (default 1).
        access: Access type -- "r" (read), "w" (write), or "rw" (both). Default "w".
    """
    try:
        session = sessions.get(session_id)
        total = session.add_watchpoint(address, size=size, access=access)
        return {"address": address, "size": size, "access": access, "total_watchpoints": total}
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def remove_watchpoint(session_id: str, address: int) -> dict:
    """Remove a memory watchpoint.

    Args:
        session_id: The session ID.
        address: The watchpoint address to remove.
    """
    try:
        session = sessions.get(session_id)
        total = session.remove_watchpoint(address)
        return {"address": address, "total_watchpoints": total}
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def list_watchpoints(session_id: str) -> dict:
    """List all memory watchpoints.

    Args:
        session_id: The session ID.
    """
    try:
        session = sessions.get(session_id)
        wps = session.list_watchpoints()
        return {"watchpoints": wps, "count": len(wps)}
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Register tools ----------------------------------------------------------


@mcp.tool()
def set_registers(session_id: str, values: dict[str, int]) -> dict:
    """Write one or more registers.

    Args:
        session_id: The session ID.
        values: Dict mapping register names to integer values.
    """
    try:
        session = sessions.get(session_id)
        updated = session.set_registers(values)
        return {"updated": updated}
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def get_registers(session_id: str, names: list[str] | None = None) -> dict:
    """Read one or more registers.

    Args:
        session_id: The session ID.
        names: List of register names. Omit or pass null for all registers.
    """
    try:
        session = sessions.get(session_id)
        regs = session.get_registers(names)
        return {"registers": regs}
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Emulation ---------------------------------------------------------------


@mcp.tool()
def emulate(
    session_id: str,
    address: int,
    stop_address: int = 0,
    count: int = 0,
    timeout_ms: int = 30000,
) -> dict:
    """Run CPU emulation.

    Must provide stop_address, count, or both to bound execution.

    Args:
        session_id: The session ID.
        address: Address to begin execution.
        stop_address: Address to stop at (exclusive).
        count: Maximum instructions to execute (capped at 100,000).
        timeout_ms: Timeout in milliseconds (capped at 60,000).
    """
    try:
        session = sessions.get(session_id)
        # Enforce caps.
        if count > _MAX_EMULATE_COUNT:
            count = _MAX_EMULATE_COUNT
        if timeout_ms > _MAX_TIMEOUT_MS:
            timeout_ms = _MAX_TIMEOUT_MS
        # Unicorn timeout is in microseconds.
        timeout_us = timeout_ms * 1000
        result = session.emulate(
            address=address,
            stop_address=stop_address,
            count=count,
            timeout_us=timeout_us,
        )
        return result
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def add_breakpoint(session_id: str, address: int, condition: str | None = None) -> dict:
    """Add a breakpoint at the given address.

    Idempotent — adding the same address twice is a no-op.

    Args:
        session_id: The session ID.
        address: The address to break at.
        condition: Optional condition expression (e.g. "eax == 42", "rax > 0x1000 and rcx != 0").
    """
    try:
        session = sessions.get(session_id)
        total = session.add_breakpoint(address, condition=condition)
        return {"address": address, "condition": condition, "total_breakpoints": total}
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def remove_breakpoint(session_id: str, address: int) -> dict:
    """Remove a breakpoint.

    Args:
        session_id: The session ID.
        address: The breakpoint address to remove.
    """
    try:
        session = sessions.get(session_id)
        total = session.remove_breakpoint(address)
        return {"address": address, "total_breakpoints": total}
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def list_breakpoints(session_id: str) -> dict:
    """List all breakpoints in the session.

    Args:
        session_id: The session ID.
    """
    try:
        session = sessions.get(session_id)
        bps = session.list_breakpoints()
        return {"breakpoints": bps, "count": len(bps)}
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def step(session_id: str, address: int | None = None) -> dict:
    """Execute a single instruction.

    If address is omitted, execution starts at the current program counter.

    Args:
        session_id: The session ID.
        address: Optional start address. Defaults to current PC.
    """
    try:
        session = sessions.get(session_id)
        return session.step(address=address)
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def save_context(session_id: str, label: str) -> dict:
    """Save a register snapshot under a label.

    Overwrites if the label already exists.

    Args:
        session_id: The session ID.
        label: A name for this snapshot.
    """
    try:
        session = sessions.get(session_id)
        labels = session.save_context(label)
        return {"label": label, "saved_labels": labels}
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def restore_context(session_id: str, label: str) -> dict:
    """Restore registers from a previously saved snapshot.

    Args:
        session_id: The session ID.
        label: The snapshot label to restore.
    """
    try:
        session = sessions.get(session_id)
        session.restore_context(label)
        return {"label": label, "registers": session.get_registers()}
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Trace tools ---------------------------------------------------------------


@mcp.tool()
def enable_trace(session_id: str, max_entries: int = 10000) -> dict:
    """Enable execution tracing.

    Clears any existing trace log and starts recording.

    Args:
        session_id: The session ID.
        max_entries: Maximum trace entries to record (default 10000).
    """
    try:
        session = sessions.get(session_id)
        session.enable_trace(max_entries=max_entries)
        return {"enabled": True, "max_entries": max_entries}
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def disable_trace(session_id: str) -> dict:
    """Disable execution tracing.

    The trace log is preserved for inspection via get_trace.

    Args:
        session_id: The session ID.
    """
    try:
        session = sessions.get(session_id)
        count = session.disable_trace()
        return {"enabled": False, "entries": count}
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def get_trace(session_id: str, offset: int = 0, limit: int = 100) -> dict:
    """Get trace entries with pagination.

    Each entry includes disassembled instruction details.

    Args:
        session_id: The session ID.
        offset: Start index (default 0).
        limit: Max entries to return (default 100).
    """
    try:
        session = sessions.get(session_id)
        return session.get_trace(offset=offset, limit=limit)
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Trace diff tools (Feature 8) -------------------------------------------


@mcp.tool()
def save_trace(session_id: str, label: str) -> dict:
    """Save the current trace log under a label.

    Overwrites if the label already exists.

    Args:
        session_id: The session ID.
        label: A name for this saved trace.
    """
    try:
        session = sessions.get(session_id)
        return session.save_trace(label)
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def diff_trace(session_id: str, label_a: str, label_b: str) -> dict:
    """Compare two saved traces instruction-by-instruction.

    Returns the common prefix length, divergence point, and up to 50 differing entries.

    Args:
        session_id: The session ID.
        label_a: First trace label.
        label_b: Second trace label.
    """
    try:
        session = sessions.get(session_id)
        return session.diff_trace(label_a, label_b)
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Syscall hooking tools (Feature 4) ---------------------------------------


@mcp.tool()
def hook_syscall(session_id: str, mode: str = "skip", default_return: int = 0) -> dict:
    """Install a syscall hook to intercept system calls.

    Modes:
        skip: Log the syscall and return default_return (continue execution).
        stop: Log the syscall and stop emulation.

    Idempotent — replaces existing hook.

    Args:
        session_id: The session ID.
        mode: Hook mode — "skip" (default) or "stop".
        default_return: Return value for skip mode (default 0).
    """
    try:
        session = sessions.get(session_id)
        return session.hook_syscall(mode=mode, default_return=default_return)
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def unhook_syscall(session_id: str) -> dict:
    """Remove the syscall hook.

    Args:
        session_id: The session ID.
    """
    try:
        session = sessions.get(session_id)
        return session.unhook_syscall()
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def get_syscall_log(session_id: str, offset: int = 0, limit: int = 100) -> dict:
    """Get recorded syscall invocations with pagination.

    Args:
        session_id: The session ID.
        offset: Start index (default 0).
        limit: Max entries to return (default 100).
    """
    try:
        session = sessions.get(session_id)
        return session.get_syscall_log(offset=offset, limit=limit)
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Symbol tools ---------------------------------------------------------------


@mcp.tool()
def add_symbol(session_id: str, name: str, address: int) -> dict:
    """Associate a symbolic name with a memory address.

    Overwrites if the name already exists.

    Args:
        session_id: The session ID.
        name: Symbol name.
        address: Memory address.
    """
    try:
        session = sessions.get(session_id)
        total = session.add_symbol(name, address)
        return {"name": name, "address": address, "total_symbols": total}
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def remove_symbol(session_id: str, name: str) -> dict:
    """Remove a symbol.

    Args:
        session_id: The session ID.
        name: The symbol name to remove.
    """
    try:
        session = sessions.get(session_id)
        total = session.remove_symbol(name)
        return {"name": name, "total_symbols": total}
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def list_symbols(session_id: str) -> dict:
    """List all symbols.

    Args:
        session_id: The session ID.
    """
    try:
        session = sessions.get(session_id)
        syms = session.list_symbols()
        return {"symbols": syms, "count": len(syms)}
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Load tools -----------------------------------------------------------------


@mcp.tool()
def load_binary(
    session_id: str, data: str, address: int,
    entry_point: int | None = None, encoding: str = "hex",
) -> dict:
    """Load binary data into the emulator.

    Auto-maps memory, writes data, and optionally sets the program counter.

    Args:
        session_id: The session ID.
        data: Binary data as hex string or base64.
        address: Destination address.
        entry_point: Optional address to set the PC to.
        encoding: "hex" (default) or "base64".
    """
    try:
        session = sessions.get(session_id)
        raw = _decode_data(data, encoding)
        return session.load_binary(raw, address, entry_point=entry_point)
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def load_executable(
    session_id: str, data: str, base_address: int = 0, encoding: str = "hex",
) -> dict:
    """Load an executable binary (ELF, PE, or Mach-O) into the emulator.

    Auto-detects format. Maps segments with correct permissions,
    sets PC to entry point, registers symbols.

    Args:
        session_id: The session ID.
        data: Binary data as hex string or base64.
        base_address: Optional base address offset. Default 0.
        encoding: "hex" (default) or "base64".
    """
    try:
        session = sessions.get(session_id)
        raw = _decode_data(data, encoding)
        return session.load_executable(raw, base_address=base_address)
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Session serialization tools (Feature 9) ---------------------------------


@mcp.tool()
def export_session(session_id: str) -> dict:
    """Export full session state (memory, registers, breakpoints, symbols) to JSON.

    Args:
        session_id: The session ID.
    """
    try:
        session = sessions.get(session_id)
        return session.export_state()
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def import_session(arch: str, state: dict) -> dict:
    """Import session state into a new session.

    Creates a new session for the given architecture and restores state from export.

    Args:
        arch: Architecture name (must match the exported state's arch).
        state: The state dict from export_session.
    """
    try:
        session = sessions.create(arch)
        session.import_state(state)
        return {
            "session_id": session.id,
            "arch": session.arch.name,
            "regions_restored": len(state.get("regions", [])),
            "breakpoints_restored": len(state.get("breakpoints", [])),
            "watchpoints_restored": len(state.get("watchpoints", [])),
            "symbols_restored": len(state.get("symbols", [])),
        }
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Standalone tools (no session needed) ------------------------------------


@mcp.tool()
def assemble(arch: str, code: str, address: int = 0) -> dict:
    """Assemble instructions into machine code using Keystone.

    Args:
        arch: Architecture name. One of: x86_32, x86_64, arm, arm64, mips32, mips32be, riscv32, riscv64.
        code: Assembly source code (e.g. "mov eax, 42; ret").
        address: Base address for assembly (affects relative offsets). Default 0.
    """
    try:
        arch_cfg = get_arch(arch)
        if arch_cfg.ks_arch is None:
            return _error(f"Assembly is not supported for {arch} (no Keystone backend)")
        ks = Ks(arch_cfg.ks_arch, arch_cfg.ks_mode)
        encoding, statement_count = ks.asm(code, addr=address)
        if encoding is None:
            return _error("Assembly produced no output")
        raw = bytes(encoding)
        return {
            "bytes_hex": raw.hex(),
            "byte_count": len(raw),
            "statement_count": statement_count,
        }
    except KsError as exc:
        return _error("Assembly failed", detail=str(exc))
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def disassemble(
    arch: str, data: str, address: int = 0, encoding: str = "hex", count: int = 0
) -> dict:
    """Disassemble machine code into instructions using Capstone.

    Args:
        arch: Architecture name. One of: x86_32, x86_64, arm, arm64, mips32, mips32be, riscv32, riscv64.
        data: Machine code as hex string or base64.
        address: Base address for disassembly. Default 0.
        encoding: "hex" (default) or "base64".
        count: Max instructions to disassemble. 0 = all.
    """
    try:
        arch_cfg = get_arch(arch)
        raw = _decode_data(data, encoding)
        cs = Cs(arch_cfg.cs_arch, arch_cfg.cs_mode)
        instructions = []
        for insn in cs.disasm(raw, address, count=count):
            instructions.append(
                {
                    "address": insn.address,
                    "mnemonic": insn.mnemonic,
                    "op_str": insn.op_str,
                    "bytes_hex": insn.bytes.hex(),
                    "size": insn.size,
                }
            )
        return {"instructions": instructions}
    except Exception as exc:
        return _error(_exc_message(exc))



# -- Memory management tools -------------------------------------------------


@mcp.tool()
def unmap_memory(session_id: str, address: int, size: int) -> dict:
    """Unmap a memory region.

    Address and size must match an existing mapping.

    Args:
        session_id: The session ID.
        address: Start address (must be page-aligned).
        size: Region size in bytes.
    """
    try:
        session = sessions.get(session_id)
        return session.unmap_memory(address, size)
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def protect_memory(session_id: str, address: int, size: int, perms: str = "rwx") -> dict:
    """Change permissions on an existing memory region.

    Args:
        session_id: The session ID.
        address: Start address (must be page-aligned).
        size: Region size in bytes.
        perms: Permission string combining 'r', 'w', 'x'. Default "rwx".
    """
    try:
        session = sessions.get(session_id)
        perm_bits = parse_perms(perms)
        return session.protect_memory(address, size, perm_bits)
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Code coverage tools -----------------------------------------------------


@mcp.tool()
def enable_coverage(session_id: str) -> dict:
    """Enable basic-block level code coverage tracking.

    Clears any existing coverage data and starts recording.

    Args:
        session_id: The session ID.
    """
    try:
        session = sessions.get(session_id)
        return session.enable_coverage()
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def disable_coverage(session_id: str) -> dict:
    """Disable code coverage tracking.

    The coverage data is preserved for inspection via get_coverage.

    Args:
        session_id: The session ID.
    """
    try:
        session = sessions.get(session_id)
        return session.disable_coverage()
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def get_coverage(session_id: str, offset: int = 0, limit: int = 100) -> dict:
    """Get collected code coverage data.

    Returns list of basic blocks hit with execution counts.

    Args:
        session_id: The session ID.
        offset: Start index (default 0).
        limit: Max entries to return (default 100).
    """
    try:
        session = sessions.get(session_id)
        return session.get_coverage(offset=offset, limit=limit)
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Convenience tools -------------------------------------------------------


@mcp.tool()
def setup_stack(session_id: str, address: int = 0x7FFF0000, size: int = 0x10000) -> dict:
    """Map a stack region and set SP to the top.

    For descending-stack architectures (all supported architectures).

    Args:
        session_id: The session ID.
        address: Stack region base address. Default 0x7FFF0000.
        size: Stack size in bytes. Default 0x10000 (64KB).
    """
    try:
        session = sessions.get(session_id)
        return session.setup_stack(address, size)
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def assemble_and_load(session_id: str, code: str, address: int) -> dict:
    """Assemble instructions and load them into the session in one call.

    Sets PC to the given address.

    Args:
        session_id: The session ID.
        code: Assembly source code.
        address: Base address for assembly and loading.
    """
    try:
        session = sessions.get(session_id)
        return session.assemble_and_load(code, address)
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def diff_context(session_id: str, label_a: str, label_b: str) -> dict:
    """Compare two saved register contexts.

    Returns only the registers that differ between the two snapshots.

    Args:
        session_id: The session ID.
        label_a: First snapshot label.
        label_b: Second snapshot label.
    """
    try:
        session = sessions.get(session_id)
        return session.diff_context(label_a, label_b)
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def fill_memory(session_id: str, address: int, size: int, pattern: str = "00") -> dict:
    """Fill a memory range with a repeating byte pattern.

    Args:
        session_id: The session ID.
        address: Start address.
        size: Number of bytes to fill.
        pattern: Hex string of bytes to repeat (e.g. "deadbeef"). Default "00".
    """
    try:
        session = sessions.get(session_id)
        raw = _decode_data(pattern, "hex")
        bytes_written = session.fill_memory(address, size, raw)
        return {"address": address, "size": bytes_written, "pattern_hex": pattern}
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def nop_out(session_id: str, address: int, size: int) -> dict:
    """Patch a memory range with architecture-appropriate NOP instructions.

    Size must be a multiple of the architecture's NOP instruction size.

    Args:
        session_id: The session ID.
        address: Start address.
        size: Number of bytes to NOP out.
    """
    try:
        session = sessions.get(session_id)
        return session.nop_out(address, size)
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def detect_arch(data: str, encoding: str = "hex") -> dict:
    """Detect the architecture of an executable binary (ELF, PE, or Mach-O).

    Returns the MCPEmulate architecture name that should be used to create a session.
    No session required.

    Args:
        data: Binary data as hex string or base64.
        encoding: \"hex\" (default) or \"base64\".
    """
    try:
        import lief
        raw = _decode_data(data, encoding)
        binary = lief.parse(list(raw))
        if binary is None:
            return _error("Failed to parse binary")

        detected_arch = None
        fmt = "unknown"
        endian = "little"

        if isinstance(binary, lief.ELF.Binary):
            fmt = "elf"
            machine = binary.header.machine_type
            is_64 = binary.header.identity_class == lief.ELF.Header.CLASS.ELF64
            endian_val = binary.header.identity_data
            endian = "big" if endian_val == lief.ELF.Header.ELF_DATA.MSB else "little"
            _elf_map = {
                lief.ELF.ARCH.I386: "x86_32",
                lief.ELF.ARCH.X86_64: "x86_64",
                lief.ELF.ARCH.ARM: "arm",
                lief.ELF.ARCH.AARCH64: "arm64",
                lief.ELF.ARCH.MIPS: "mips32be" if endian == "big" else "mips32",
                lief.ELF.ARCH.RISCV: "riscv64" if is_64 else "riscv32",
            }
            detected_arch = _elf_map.get(machine)

        elif isinstance(binary, lief.PE.Binary):
            fmt = "pe"
            machine = binary.header.machine
            _pe_map = {
                lief.PE.Header.MACHINE_TYPES.I386: "x86_32",
                lief.PE.Header.MACHINE_TYPES.AMD64: "x86_64",
                lief.PE.Header.MACHINE_TYPES.ARM: "arm",
                lief.PE.Header.MACHINE_TYPES.ARM64: "arm64",
            }
            detected_arch = _pe_map.get(machine)

        elif isinstance(binary, lief.MachO.Binary):
            fmt = "macho"
            cpu_type = binary.header.cpu_type
            _macho_map = {
                lief.MachO.Header.CPU_TYPE.x86: "x86_32",
                lief.MachO.Header.CPU_TYPE.x86_64: "x86_64",
                lief.MachO.Header.CPU_TYPE.ARM: "arm",
                lief.MachO.Header.CPU_TYPE.ARM64: "arm64",
            }
            detected_arch = _macho_map.get(cpu_type)

        if detected_arch is None:
            return _error(f"Could not determine architecture from {fmt} binary")

        return {"arch": detected_arch, "format": fmt, "endian": endian}
    except Exception as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def run_and_diff(
    session_id: str, address: int, stop_address: int = 0,
    count: int = 0, timeout_ms: int = 30000,
) -> dict:
    """Snapshot state, emulate, and return combined register and memory diffs.

    Equivalent to snapshot + emulate + snapshot + diff in one call.

    Args:
        session_id: The session ID.
        address: Address to begin execution.
        stop_address: Address to stop at (exclusive). Default 0.
        count: Maximum instructions to execute. Default 0 (unlimited).
        timeout_ms: Timeout in milliseconds (max 60000). Default 30000.
    """
    try:
        session = sessions.get(session_id)
        timeout_ms = min(max(timeout_ms, 0), _MAX_TIMEOUT_MS)
        timeout_us = timeout_ms * 1000
        return session.run_and_diff(
            address=address, stop_address=stop_address,
            count=count, timeout_us=timeout_us,
        )
    except Exception as exc:
        return _error(_exc_message(exc))


# -- Entry point -------------------------------------------------------------


def main() -> None:
    """Run the MCP server. Defaults to stdio; supports --transport sse|streamable-http."""
    import argparse
    parser = argparse.ArgumentParser(prog="mcp-emulate", description="MCP CPU emulation server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse", "streamable-http"],
        default="stdio", help="Transport protocol (default: stdio)",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
