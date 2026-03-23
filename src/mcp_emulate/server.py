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
        return _error(str(exc))


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
        return _error(str(exc))


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
        return {
            "address": region.address,
            "size": region.size,
            "perms": _perms_str(region.perms),
        }
    except (KeyError, ValueError, Exception) as exc:
        return _error(str(exc))


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
        return _error(str(exc))


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
        return _error(str(exc))


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
        return _error(str(exc))


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
        return _error(str(exc))


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
        return _error(str(exc))


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
        return _error(str(exc))


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
        return _error(str(exc))


# -- Entry point -------------------------------------------------------------


def main() -> None:
    """Run the MCP server on stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
