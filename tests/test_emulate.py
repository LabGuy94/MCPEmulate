"""Integration tests for MCPEmulate — full assemble-load-run-inspect round trip."""

from __future__ import annotations

import pytest

from mcp_emulate.architectures import get_arch, ARCHITECTURES
from mcp_emulate.session import SessionManager, parse_perms, align_up


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def manager() -> SessionManager:
    return SessionManager()


# ---------------------------------------------------------------------------
# Architecture lookup
# ---------------------------------------------------------------------------

class TestArchitectures:
    def test_valid_arches(self) -> None:
        for name in ("x86_32", "x86_64", "arm", "arm64"):
            cfg = get_arch(name)
            assert cfg.name == name
            assert cfg.pc_reg in cfg.register_map
            assert cfg.sp_reg in cfg.register_map

    def test_invalid_arch_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown architecture"):
            get_arch("pdp11")


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

class TestSessionManager:
    def test_create_and_destroy(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        assert session.id
        assert session.arch.name == "x86_32"
        manager.destroy(session.id)
        with pytest.raises(KeyError):
            manager.get(session.id)

    def test_destroy_nonexistent_raises(self, manager: SessionManager) -> None:
        with pytest.raises(KeyError):
            manager.destroy("does-not-exist")

    def test_max_sessions(self, manager: SessionManager) -> None:
        ids = []
        for _ in range(SessionManager.MAX_SESSIONS):
            s = manager.create("x86_32")
            ids.append(s.id)
        with pytest.raises(RuntimeError, match="Maximum number of sessions"):
            manager.create("x86_32")
        # Cleanup.
        for sid in ids:
            manager.destroy(sid)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtils:
    def test_parse_perms_rwx(self) -> None:
        from unicorn import UC_PROT_ALL
        assert parse_perms("rwx") == UC_PROT_ALL

    def test_parse_perms_rx(self) -> None:
        from unicorn import UC_PROT_READ, UC_PROT_EXEC
        assert parse_perms("rx") == UC_PROT_READ | UC_PROT_EXEC

    def test_parse_perms_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid permission character"):
            parse_perms("z")

    def test_parse_perms_empty(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            parse_perms("")

    def test_align_up(self) -> None:
        assert align_up(1, 0x1000) == 0x1000
        assert align_up(0x1000, 0x1000) == 0x1000
        assert align_up(0x1001, 0x1000) == 0x2000
        assert align_up(0, 0x1000) == 0


# ---------------------------------------------------------------------------
# Memory operations
# ---------------------------------------------------------------------------

class TestMemory:
    def test_map_and_write_read(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        session.write_memory(0x1000, b"\x90\xc3")
        data = session.read_memory(0x1000, 2)
        assert data == b"\x90\xc3"

    def test_size_alignment(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        region = session.map_memory(0x2000, 1)  # 1 byte -> rounded to 4KB
        assert region.size == 0x1000

    def test_write_unmapped_raises(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        with pytest.raises(Exception):
            session.write_memory(0xDEAD0000, b"\x90")


# ---------------------------------------------------------------------------
# Register operations
# ---------------------------------------------------------------------------

class TestRegisters:
    def test_set_get_single(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.set_register("eax", 42)
        assert session.get_register("eax") == 42

    def test_bulk_set_get(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.set_registers({"eax": 1, "ebx": 2, "ecx": 3})
        regs = session.get_registers(["eax", "ebx", "ecx"])
        assert regs == {"eax": 1, "ebx": 2, "ecx": 3}

    def test_get_all(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        all_regs = session.get_registers()
        assert "eax" in all_regs
        assert "eip" in all_regs

    def test_invalid_register_raises(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        with pytest.raises(ValueError, match="Unknown register"):
            session.set_register("xmm99", 0)


# ---------------------------------------------------------------------------
# Emulation: full round-trip
# ---------------------------------------------------------------------------

class TestEmulation:
    def test_x86_32_mov_eax_42(self, manager: SessionManager) -> None:
        """Full round-trip: assemble -> load -> run -> inspect."""
        from keystone import Ks
        from capstone import Cs

        arch = get_arch("x86_32")
        session = manager.create("x86_32")

        # 1. Map code and stack regions.
        session.map_memory(0x1000, 0x1000)    # code
        session.map_memory(0x100000, 0x1000)   # stack

        # 2. Assemble "mov eax, 42; ret".
        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, count = ks.asm("mov eax, 42; ret")
        assert encoding is not None
        code_bytes = bytes(encoding)

        # 3. Write code to memory.
        session.write_memory(0x1000, code_bytes)

        # 4. Set up stack: ESP inside mapped page, write a return address
        #    that we use as stop_address so ret completes cleanly.
        stop_addr = 0xDEAD
        session.map_memory(0xD000, 0x1000)  # map page containing stop_addr
        session.set_register("esp", 0x100FFC)  # within stack page
        # Push return address onto stack (little-endian).
        session.write_memory(0x100FFC, stop_addr.to_bytes(4, "little"))

        # 5. Emulate until ret jumps to stop_addr.
        result = session.emulate(address=0x1000, stop_address=stop_addr, count=100)

        # 6. Verify eax == 42.
        eax = session.get_register("eax")
        assert eax == 42, f"Expected eax=42, got {eax}"

        # 7. Check emulation completed cleanly.
        assert result["stop_reason"] == "completed", f"Got: {result}"
        assert result["instructions_executed"] == 2  # mov + ret

        # 8. Disassemble the code for verification.
        cs = Cs(arch.cs_arch, arch.cs_mode)
        instructions = list(cs.disasm(code_bytes, 0x1000))
        assert len(instructions) >= 2
        assert instructions[0].mnemonic == "mov"

    def test_emulate_requires_bound(self, manager: SessionManager) -> None:
        """Emulate must be given stop_address or count."""
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        with pytest.raises(ValueError, match="Must provide"):
            session.emulate(address=0x1000)

    def test_x86_64_mov_rax(self, manager: SessionManager) -> None:
        """Verify x86_64 works end to end."""
        from keystone import Ks

        arch = get_arch("x86_64")
        session = manager.create("x86_64")

        session.map_memory(0x1000, 0x1000)    # code
        session.map_memory(0x200000, 0x1000)   # stack
        session.map_memory(0xD000, 0x1000)     # page for stop address

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov rax, 0xDEADBEEF; ret")
        code_bytes = bytes(encoding)

        session.write_memory(0x1000, code_bytes)
        stop_addr = 0xDEAD
        session.set_register("rsp", 0x200FF8)  # within stack page, 8-byte aligned for x64
        session.write_memory(0x200FF8, stop_addr.to_bytes(8, "little"))

        result = session.emulate(address=0x1000, stop_address=stop_addr, count=100)
        assert result["stop_reason"] == "completed", f"Got: {result}"
        rax = session.get_register("rax")
        assert rax == 0xDEADBEEF


# ---------------------------------------------------------------------------
# Server tool wrappers (import-level smoke test)
# ---------------------------------------------------------------------------

class TestServerTools:
    """Smoke-test the tool functions directly (no MCP protocol)."""

    def test_assemble_disassemble_roundtrip(self) -> None:
        from mcp_emulate.server import assemble, disassemble

        # Assemble.
        asm_result = assemble(arch="x86_32", code="nop; nop; ret")
        assert "error" not in asm_result
        assert asm_result["byte_count"] >= 3

        # Disassemble what we assembled.
        dis_result = disassemble(
            arch="x86_32", data=asm_result["bytes_hex"]
        )
        assert "error" not in dis_result
        mnemonics = [i["mnemonic"] for i in dis_result["instructions"]]
        assert "nop" in mnemonics
        assert "ret" in mnemonics

    def test_assemble_invalid_code(self) -> None:
        from mcp_emulate.server import assemble

        result = assemble(arch="x86_32", code="this_is_not_assembly xyz")
        assert "error" in result

    def test_assemble_invalid_arch(self) -> None:
        from mcp_emulate.server import assemble

        result = assemble(arch="z80", code="nop")
        assert "error" in result

    def test_create_destroy_roundtrip(self) -> None:
        from mcp_emulate.server import create_emulator, destroy_emulator

        result = create_emulator(arch="x86_32")
        assert "session_id" in result
        sid = result["session_id"]

        destroy_result = destroy_emulator(session_id=sid)
        assert destroy_result.get("success") is True

    def test_full_tool_roundtrip(self) -> None:
        """Exercise the tool-level API for a full cycle."""
        from mcp_emulate.server import (
            create_emulator,
            destroy_emulator,
            map_memory,
            write_memory,
            read_memory,
            set_registers,
            get_registers,
            emulate,
            assemble,
        )

        # Create session.
        r = create_emulator(arch="x86_32")
        assert "error" not in r
        sid = r["session_id"]

        # Map memory.
        r = map_memory(session_id=sid, address=0x1000, size=0x1000)
        assert "error" not in r
        r = map_memory(session_id=sid, address=0x100000, size=0x1000)
        assert "error" not in r
        # Page for stop address.
        r = map_memory(session_id=sid, address=0xD000, size=0x1000)
        assert "error" not in r

        # Assemble and write code.
        r = assemble(arch="x86_32", code="mov eax, 99; ret", address=0x1000)
        assert "error" not in r
        code_hex = r["bytes_hex"]

        r = write_memory(session_id=sid, address=0x1000, data=code_hex)
        assert "error" not in r

        # Set up stack with return address.
        stop_addr = 0xDEAD
        r = set_registers(session_id=sid, values={"esp": 0x100FFC})
        assert "error" not in r
        # Write return address (little-endian) at ESP.
        r = write_memory(
            session_id=sid, address=0x100FFC,
            data=stop_addr.to_bytes(4, "little").hex()
        )
        assert "error" not in r

        # Run.
        r = emulate(session_id=sid, address=0x1000, stop_address=stop_addr, count=100)
        assert "error" not in r
        assert r["stop_reason"] == "completed", f"Got: {r}"
        assert r["instructions_executed"] == 2

        # Check register.
        r = get_registers(session_id=sid, names=["eax"])
        assert "error" not in r
        assert r["registers"]["eax"] == 99

        # Read memory back.
        r = read_memory(session_id=sid, address=0x1000, size=len(code_hex) // 2)
        assert "error" not in r
        assert r["data"] == code_hex

        # Cleanup.
        r = destroy_emulator(session_id=sid)
        assert r.get("success") is True
