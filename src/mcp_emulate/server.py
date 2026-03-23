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
        arch: Architecture name. One of: x86_32, x86_64, arm, arm64.

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
    except (KeyError, ValueError, Exception) as exc:
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
    except (KeyError, ValueError, Exception) as exc:
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
    except (KeyError, ValueError, Exception) as exc:
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
    except (KeyError, Exception) as exc:
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
    except (KeyError, Exception) as exc:
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
    except (KeyError, ValueError, Exception) as exc:
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
    except (KeyError, ValueError, Exception) as exc:
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
    except (KeyError, Exception) as exc:
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
    except (KeyError, Exception) as exc:
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
    except (KeyError, ValueError, Exception) as exc:
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
    except (KeyError, ValueError, Exception) as exc:
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
    except (KeyError, ValueError, Exception) as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def add_breakpoint(session_id: str, address: int) -> dict:
    """Add a breakpoint at the given address.

    Idempotent \u2014 adding the same address twice is a no-op.

    Args:
        session_id: The session ID.
        address: The address to break at.
    """
    try:
        session = sessions.get(session_id)
        total = session.add_breakpoint(address)
        return {"address": address, "total_breakpoints": total}
    except (KeyError, Exception) as exc:
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
    except (KeyError, Exception) as exc:
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
    except (KeyError, Exception) as exc:
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
    except (KeyError, ValueError, Exception) as exc:
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
    except (KeyError, Exception) as exc:
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
    except (KeyError, Exception) as exc:
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
    except (KeyError, Exception) as exc:
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
    except (KeyError, Exception) as exc:
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
    except (KeyError, Exception) as exc:
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
    except (KeyError, Exception) as exc:
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
    except (KeyError, Exception) as exc:
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
    except (KeyError, Exception) as exc:
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
    except (KeyError, ValueError, Exception) as exc:
        return _error(_exc_message(exc))



# -- Standalone tools (no session needed) ------------------------------------


@mcp.tool()
def assemble(arch: str, code: str, address: int = 0) -> dict:
    """Assemble instructions into machine code using Keystone.

    Args:
        arch: Architecture name (x86_32, x86_64, arm, arm64).
        code: Assembly source code (e.g. "mov eax, 42; ret").
        address: Base address for assembly (affects relative offsets). Default 0.
    """
    try:
        arch_cfg = get_arch(arch)
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
    except (ValueError, Exception) as exc:
        return _error(_exc_message(exc))


@mcp.tool()
def disassemble(
    arch: str, data: str, address: int = 0, encoding: str = "hex", count: int = 0
) -> dict:
    """Disassemble machine code into instructions using Capstone.

    Args:
        arch: Architecture name (x86_32, x86_64, arm, arm64).
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
    except (ValueError, Exception) as exc:
        return _error(_exc_message(exc))


# -- Entry point -------------------------------------------------------------


def main() -> None:
    """Run the MCP server on stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
