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



# ---------------------------------------------------------------------------
# Breakpoints
# ---------------------------------------------------------------------------

class TestBreakpoints:
    def test_add_remove_breakpoint(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        assert session.add_breakpoint(0x1000) == 1
        assert session.add_breakpoint(0x2000) == 2
        assert session.list_breakpoints() == [0x1000, 0x2000]
        assert session.remove_breakpoint(0x1000) == 1
        assert session.list_breakpoints() == [0x2000]

    def test_add_duplicate_is_idempotent(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        assert session.add_breakpoint(0x1000) == 1
        assert session.add_breakpoint(0x1000) == 1  # no change
        assert session.list_breakpoints() == [0x1000]

    def test_remove_nonexistent_raises(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        with pytest.raises(KeyError, match="No breakpoint"):
            session.remove_breakpoint(0xDEAD)

    def test_emulate_hits_breakpoint(self, manager: SessionManager) -> None:
        """Assemble 4 MOV instructions, breakpoint on 3rd, verify stop."""
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        # 4 instructions: each 'mov reg, imm32' is 5 bytes on x86_32
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2; mov ecx, 3; mov edx, 4")
        code = bytes(encoding)
        session.write_memory(0x1000, code)

        # Breakpoint on 3rd instruction (offset 10 = 2 * 5 bytes)
        bp_addr = 0x1000 + 10
        session.add_breakpoint(bp_addr)

        result = session.emulate(address=0x1000, count=100)
        assert result["stop_reason"] == "breakpoint"
        assert result["breakpoint_address"] == bp_addr
        # Only first 2 instructions executed
        assert result["instructions_executed"] == 2
        assert result["registers"]["eax"] == 1
        assert result["registers"]["ebx"] == 2
        # 3rd and 4th did NOT execute
        assert result["registers"]["ecx"] == 0
        assert result["registers"]["edx"] == 0
        # PC should be at the breakpoint address
        assert result["registers"]["eip"] == bp_addr

    def test_resume_after_breakpoint(self, manager: SessionManager) -> None:
        """Hit a breakpoint, then resume and verify remaining instructions execute."""
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2; mov ecx, 3; mov edx, 4")
        code = bytes(encoding)
        session.write_memory(0x1000, code)

        # Breakpoint on 3rd instruction
        bp_addr = 0x1000 + 10
        session.add_breakpoint(bp_addr)

        # First run: hits breakpoint
        result = session.emulate(address=0x1000, count=100)
        assert result["stop_reason"] == "breakpoint"

        # Remove breakpoint and resume from PC
        session.remove_breakpoint(bp_addr)
        pc = result["registers"]["eip"]
        result2 = session.emulate(address=pc, count=2)
        assert result2["stop_reason"] == "count_exhausted"
        assert result2["registers"]["ecx"] == 3
        assert result2["registers"]["edx"] == 4


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_single_instruction(self, manager: SessionManager) -> None:
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 42; mov ebx, 99")
        session.write_memory(0x1000, bytes(encoding))

        result = session.step(address=0x1000)
        assert result["address"] == 0x1000
        assert result["instruction"]["mnemonic"] == "mov"
        assert result["registers"]["eax"] == 42
        # ebx should not have been touched yet
        assert result["registers"]["ebx"] == 0

    def test_step_sequence(self, manager: SessionManager) -> None:
        """Step 3 times and verify progressive register changes."""
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2; mov ecx, 3")
        session.write_memory(0x1000, bytes(encoding))

        r1 = session.step(address=0x1000)
        assert r1["registers"]["eax"] == 1
        assert r1["registers"]["ebx"] == 0

        # Step from where PC now points
        pc = r1["registers"]["eip"]
        r2 = session.step(address=pc)
        assert r2["registers"]["ebx"] == 2
        assert r2["registers"]["ecx"] == 0

        pc = r2["registers"]["eip"]
        r3 = session.step(address=pc)
        assert r3["registers"]["ecx"] == 3

    def test_step_from_explicit_address(self, manager: SessionManager) -> None:
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 10; mov ebx, 20")
        code = bytes(encoding)
        session.write_memory(0x1000, code)

        # Skip first instruction, step only the second
        second_addr = 0x1000 + 5  # mov eax, imm32 is 5 bytes
        result = session.step(address=second_addr)
        assert result["address"] == second_addr
        assert result["registers"]["ebx"] == 20
        # eax was never executed
        assert result["registers"]["eax"] == 0


# ---------------------------------------------------------------------------
# Context save/restore
# ---------------------------------------------------------------------------

class TestContext:
    def test_save_restore_roundtrip(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.set_registers({"eax": 100, "ebx": 200})
        labels = session.save_context("snap1")
        assert "snap1" in labels

        # Change registers
        session.set_registers({"eax": 999, "ebx": 888})
        assert session.get_register("eax") == 999

        # Restore
        session.restore_context("snap1")
        assert session.get_register("eax") == 100
        assert session.get_register("ebx") == 200

    def test_restore_nonexistent_raises(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        with pytest.raises(KeyError, match="No saved context"):
            session.restore_context("nope")

    def test_overwrite_label(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.set_register("eax", 10)
        session.save_context("snap")

        session.set_register("eax", 20)
        session.save_context("snap")  # overwrite

        session.set_register("eax", 99)
        session.restore_context("snap")
        assert session.get_register("eax") == 20  # gets the latest save

    def test_multiple_labels(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.set_register("eax", 1)
        session.save_context("a")

        session.set_register("eax", 2)
        session.save_context("b")

        session.set_register("eax", 99)

        session.restore_context("a")
        assert session.get_register("eax") == 1

        session.restore_context("b")
        assert session.get_register("eax") == 2


# ---------------------------------------------------------------------------
# Server tool wrappers for new tools
# ---------------------------------------------------------------------------

class TestServerBreakpointTools:
    def test_breakpoint_tool_roundtrip(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory,
            add_breakpoint, remove_breakpoint, list_breakpoints,
        )

        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)

        r = add_breakpoint(session_id=sid, address=0x1000)
        assert "error" not in r
        assert r["total_breakpoints"] == 1

        r = list_breakpoints(session_id=sid)
        assert r["breakpoints"] == [0x1000]
        assert r["count"] == 1

        r = remove_breakpoint(session_id=sid, address=0x1000)
        assert "error" not in r
        assert r["total_breakpoints"] == 0

        r = list_breakpoints(session_id=sid)
        assert r["count"] == 0

        destroy_emulator(session_id=sid)

    def test_step_tool(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory,
            write_memory, assemble, step,
        )

        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)

        asm = assemble(arch="x86_32", code="mov eax, 77", address=0x1000)
        write_memory(session_id=sid, address=0x1000, data=asm["bytes_hex"])

        r = step(session_id=sid, address=0x1000)
        assert "error" not in r
        assert r["instruction"]["mnemonic"] == "mov"
        assert r["registers"]["eax"] == 77

        destroy_emulator(session_id=sid)

    def test_context_tool_roundtrip(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator,
            set_registers, get_registers,
            save_context, restore_context,
        )

        r = create_emulator(arch="x86_32")
        sid = r["session_id"]

        set_registers(session_id=sid, values={"eax": 42})
        r = save_context(session_id=sid, label="test")
        assert "error" not in r
        assert "test" in r["saved_labels"]

        set_registers(session_id=sid, values={"eax": 0})
        r = restore_context(session_id=sid, label="test")
        assert "error" not in r
        assert r["registers"]["eax"] == 42

        destroy_emulator(session_id=sid)


# ---------------------------------------------------------------------------
# Memory inspection (Iteration 3)
# ---------------------------------------------------------------------------

class TestMemoryInspection:
    def test_list_regions(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        session.map_memory(0x2000, 0x2000)
        regions = session.list_regions()
        assert len(regions) == 2
        assert regions[0]["address"] == 0x1000
        assert regions[1]["address"] == 0x2000
        assert regions[1]["size"] == 0x2000

    def test_hexdump_basic(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        session.write_memory(0x1000, b"Hello World\x00")
        dump = session.hexdump(0x1000, 16)
        assert "48 65 6c 6c 6f 20 57 6f" in dump
        assert "|Hello World." in dump

    def test_hexdump_size_cap(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x2000)
        # Request more than 4096 — should be capped.
        dump = session.hexdump(0x1000, 8192)
        # 4096 bytes / 16 per line = 256 lines
        lines = dump.strip().split("\n")
        assert len(lines) == 256

    def test_search_memory_found(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        session.write_memory(0x1000, b"\x00" * 16 + b"\xDE\xAD" + b"\x00" * 32 + b"\xDE\xAD")
        matches = session.search_memory(b"\xDE\xAD")
        assert 0x1010 in matches
        assert 0x1032 in matches
        assert len(matches) == 2

    def test_search_memory_not_found(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        session.write_memory(0x1000, b"\x00" * 64)
        matches = session.search_memory(b"\xFF\xFF")
        assert matches == []


# ---------------------------------------------------------------------------
# Watchpoints (Iteration 3)
# ---------------------------------------------------------------------------

class TestWatchpoints:
    def test_add_remove_watchpoint(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        total = session.add_watchpoint(0x1000, size=4, access="w")
        assert total == 1
        wps = session.list_watchpoints()
        assert len(wps) == 1
        assert wps[0]["address"] == 0x1000
        assert wps[0]["size"] == 4
        assert wps[0]["access"] == "w"
        total = session.remove_watchpoint(0x1000)
        assert total == 0
        assert session.list_watchpoints() == []

    def test_watchpoint_on_write(self, manager: SessionManager) -> None:
        """Assemble code that writes to memory, add write watchpoint, verify stop."""
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)  # code
        session.map_memory(0x2000, 0x1000)  # data

        ks = Ks(arch.ks_arch, arch.ks_mode)
        # mov dword ptr [0x2000], 0x42  — writes to 0x2000
        encoding, _ = ks.asm("mov eax, 0x42; mov dword ptr [0x2000], eax")
        session.write_memory(0x1000, bytes(encoding))

        session.add_watchpoint(0x2000, size=4, access="w")
        result = session.emulate(address=0x1000, count=10)
        assert result["stop_reason"] == "watchpoint"
        assert result["watchpoint"]["access"] == "write"
        assert result["watchpoint"]["address"] == 0x2000

    def test_watchpoint_on_read(self, manager: SessionManager) -> None:
        """Assemble code that reads from memory, add read watchpoint, verify stop."""
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)  # code
        session.map_memory(0x2000, 0x1000)  # data
        session.write_memory(0x2000, b"\x42\x00\x00\x00")

        ks = Ks(arch.ks_arch, arch.ks_mode)
        # mov eax, dword ptr [0x2000]  — reads from 0x2000
        encoding, _ = ks.asm("mov eax, dword ptr [0x2000]")
        session.write_memory(0x1000, bytes(encoding))

        session.add_watchpoint(0x2000, size=4, access="r")
        result = session.emulate(address=0x1000, count=10)
        assert result["stop_reason"] == "watchpoint"
        assert result["watchpoint"]["access"] == "read"

    def test_watchpoint_idempotent(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        session.add_watchpoint(0x1000, size=4, access="w")
        session.add_watchpoint(0x1000, size=4, access="w")  # replace
        assert len(session.list_watchpoints()) == 1

    def test_remove_nonexistent_raises(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        with pytest.raises(KeyError, match="No watchpoint"):
            session.remove_watchpoint(0xDEAD)


# ---------------------------------------------------------------------------
# Server tools — Memory inspection & Watchpoints (Iteration 3)
# ---------------------------------------------------------------------------

class TestServerMemoryTools:
    def test_list_regions_tool(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory, list_regions,
        )

        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)
        map_memory(session_id=sid, address=0x2000, size=0x2000)

        r = list_regions(session_id=sid)
        assert "error" not in r
        assert r["count"] == 2
        assert r["regions"][0]["perms"] == "rwx"

        destroy_emulator(session_id=sid)

    def test_watchpoint_tool_roundtrip(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory,
            add_watchpoint, remove_watchpoint, list_watchpoints,
        )

        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)

        r = add_watchpoint(session_id=sid, address=0x1000, size=4, access="w")
        assert "error" not in r
        assert r["total_watchpoints"] == 1

        r = list_watchpoints(session_id=sid)
        assert r["count"] == 1
        assert r["watchpoints"][0]["address"] == 0x1000

        r = remove_watchpoint(session_id=sid, address=0x1000)
        assert "error" not in r
        assert r["total_watchpoints"] == 0

        r = list_watchpoints(session_id=sid)
        assert r["count"] == 0

        destroy_emulator(session_id=sid)