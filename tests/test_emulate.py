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
        assert session.list_breakpoints() == [{"address": 0x1000, "condition": None}, {"address": 0x2000, "condition": None}]
        assert session.remove_breakpoint(0x1000) == 1
        assert session.list_breakpoints() == [{"address": 0x2000, "condition": None}]

    def test_add_duplicate_is_idempotent(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        assert session.add_breakpoint(0x1000) == 1
        assert session.add_breakpoint(0x1000) == 1  # no change
        assert session.list_breakpoints() == [{"address": 0x1000, "condition": None}]

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


    def test_step_near_region_end(self, manager: SessionManager) -> None:
        """Step at end of a small region should not fail on disassembly read."""
        from keystone import Ks
        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        # Map only 6 bytes total.
        session.map_memory(0x1000, 0x1000)  # minimum page size
        ks = Ks(arch.ks_arch, arch.ks_mode)
        # 'nop' is 1 byte (0x90). Write one nop at offset 0xFFE (2 bytes from end).
        encoding, _ = ks.asm("nop")
        session.write_memory(0x1000 + 0xFFE, bytes(encoding))
        result = session.step(address=0x1000 + 0xFFE)
        assert result["instruction"]["mnemonic"] == "nop"

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
        assert r["breakpoints"] == [{"address": 0x1000, "condition": None}]
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
        # 4096 bytes of all-zeros collapses: first line + '*' = 2 lines.
        lines = dump.strip().split("\n")
        assert len(lines) == 2
        assert lines[1] == "*"

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


# ---------------------------------------------------------------------------
# Execution Trace (Iteration 4)
# ---------------------------------------------------------------------------

class TestTrace:
    def test_trace_basic(self, manager: SessionManager) -> None:
        """Enable trace, emulate 3 instructions, verify 3 entries with correct mnemonics."""
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2; mov ecx, 3")
        session.write_memory(0x1000, bytes(encoding))

        session.enable_trace()
        session.emulate(address=0x1000, count=3)
        trace = session.get_trace()
        assert trace["available"] == 3
        assert trace["entries"][0]["mnemonic"] == "mov"
        assert trace["entries"][0]["address"] == 0x1000
        assert trace["entries"][2]["index"] == 2

    def test_trace_disabled_by_default(self, manager: SessionManager) -> None:
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2")
        session.write_memory(0x1000, bytes(encoding))

        session.emulate(address=0x1000, count=2)
        trace = session.get_trace()
        assert trace["available"] == 0
        assert trace["entries"] == []

    def test_trace_with_breakpoint(self, manager: SessionManager) -> None:
        """Breakpointed instruction NOT in trace, preceding instructions ARE."""
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        # 3 instructions: each 'mov reg, imm32' is 5 bytes
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2; mov ecx, 3")
        session.write_memory(0x1000, bytes(encoding))

        # Breakpoint on 3rd instruction (offset 10)
        session.add_breakpoint(0x1000 + 10)
        session.enable_trace()
        result = session.emulate(address=0x1000, count=10)
        assert result["stop_reason"] == "breakpoint"

        trace = session.get_trace()
        # Only 2 instructions traced (3rd was breakpointed, not traced)
        assert trace["available"] == 2
        assert trace["entries"][0]["address"] == 0x1000
        assert trace["entries"][1]["address"] == 0x1000 + 5

    def test_trace_pagination(self, manager: SessionManager) -> None:
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2; mov ecx, 3; mov edx, 4; inc eax")
        session.write_memory(0x1000, bytes(encoding))

        session.enable_trace()
        session.emulate(address=0x1000, count=5)

        trace = session.get_trace(offset=2, limit=2)
        assert trace["available"] == 5
        assert len(trace["entries"]) == 2
        assert trace["entries"][0]["index"] == 2
        assert trace["entries"][1]["index"] == 3

    def test_trace_max_cap(self, manager: SessionManager) -> None:
        """Ring buffer keeps the LAST N entries when max_entries is exceeded."""
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2; mov ecx, 3; mov edx, 4; inc eax")
        session.write_memory(0x1000, bytes(encoding))

        session.enable_trace(max_entries=3)
        session.emulate(address=0x1000, count=5)
        trace = session.get_trace(limit=10)
        assert trace["available"] == 3  # capped at 3
        # Ring buffer keeps LAST 3 entries (instructions 3, 4, 5).
        # Instruction 3 is 'mov ecx, 3' at offset 10, 4 is 'mov edx, 4' at offset 15,
        # 5 is 'inc eax' at offset 20.
        assert trace["entries"][0]["address"] == 0x1000 + 10
        assert trace["entries"][2]["mnemonic"] == "inc"

    def test_enable_clears_previous(self, manager: SessionManager) -> None:
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2")
        session.write_memory(0x1000, bytes(encoding))

        session.enable_trace()
        session.emulate(address=0x1000, count=2)
        assert session.get_trace()["available"] == 2

        # Enable again — should clear
        session.enable_trace()
        assert session.get_trace()["available"] == 0


# ---------------------------------------------------------------------------
# Server tools — Trace (Iteration 4)
# ---------------------------------------------------------------------------

class TestServerTraceTools:
    def test_trace_tool_roundtrip(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory,
            write_memory, assemble, emulate,
            enable_trace, disable_trace, get_trace,
        )

        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)

        asm = assemble(arch="x86_32", code="mov eax, 1; mov ebx, 2", address=0x1000)
        write_memory(session_id=sid, address=0x1000, data=asm["bytes_hex"])

        r = enable_trace(session_id=sid)
        assert "error" not in r
        assert r["enabled"] is True

        emulate(session_id=sid, address=0x1000, count=2)

        r = get_trace(session_id=sid)
        assert "error" not in r
        assert r["available"] == 2
        assert r["entries"][0]["mnemonic"] == "mov"

        r = disable_trace(session_id=sid)
        assert "error" not in r
        assert r["enabled"] is False
        assert r["entries"] == 2

        destroy_emulator(session_id=sid)


# ---------------------------------------------------------------------------
# Symbols (Iteration 5)
# ---------------------------------------------------------------------------

class TestSymbols:
    def test_add_remove_symbol(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        total = session.add_symbol("main", 0x1000)
        assert total == 1
        syms = session.list_symbols()
        assert len(syms) == 1
        assert syms[0]["name"] == "main"
        assert syms[0]["address"] == 0x1000
        total = session.remove_symbol("main")
        assert total == 0
        assert session.list_symbols() == []

    def test_overwrite_symbol(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.add_symbol("entry", 0x1000)
        session.add_symbol("entry", 0x2000)  # overwrite
        syms = session.list_symbols()
        assert len(syms) == 1
        assert syms[0]["address"] == 0x2000

    def test_remove_nonexistent_raises(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        with pytest.raises(KeyError, match="No symbol"):
            session.remove_symbol("nope")

    def test_list_symbols_sorted(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.add_symbol("zebra", 0x3000)
        session.add_symbol("alpha", 0x1000)
        session.add_symbol("mid", 0x2000)
        syms = session.list_symbols()
        names = [s["name"] for s in syms]
        assert names == ["alpha", "mid", "zebra"]


# ---------------------------------------------------------------------------
# Load Binary (Iteration 5)
# ---------------------------------------------------------------------------

class TestLoadBinary:
    def test_load_binary_basic(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        code = b"\xb8\x2a\x00\x00\x00"  # mov eax, 42
        result = session.load_binary(code, address=0x1000, entry_point=0x1000)
        assert result["address"] == 0x1000
        assert result["size"] == 5
        assert result["entry_point"] == 0x1000
        # Verify memory was written.
        assert session.read_memory(0x1000, 5) == code
        # Verify PC was set.
        assert session.get_register("eip") == 0x1000

    def test_load_binary_auto_map(self, manager: SessionManager) -> None:
        """Load at address not yet mapped — auto-map should handle it."""
        session = manager.create("x86_32")
        code = b"\x90\x90\xc3"  # nop; nop; ret
        result = session.load_binary(code, address=0x5000)
        assert result["size"] == 3
        # Memory should be readable.
        assert session.read_memory(0x5000, 3) == code

    def test_load_binary_no_entry_point(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        # Set PC to a known value first.
        session.map_memory(0x1000, 0x1000)
        session.set_register("eip", 0xDEAD)
        code = b"\x90\xc3"
        result = session.load_binary(code, address=0x2000)
        assert result["entry_point"] is None
        # PC should be unchanged.
        assert session.get_register("eip") == 0xDEAD


# ---------------------------------------------------------------------------
# Server tools — Symbols & Load (Iteration 5)
# ---------------------------------------------------------------------------

class TestServerIteration5Tools:
    def test_symbol_tool_roundtrip(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator,
            add_symbol, remove_symbol, list_symbols,
        )

        r = create_emulator(arch="x86_32")
        sid = r["session_id"]

        r = add_symbol(session_id=sid, name="main", address=0x1000)
        assert "error" not in r
        assert r["total_symbols"] == 1

        r = list_symbols(session_id=sid)
        assert r["count"] == 1
        assert r["symbols"][0]["name"] == "main"

        r = remove_symbol(session_id=sid, name="main")
        assert "error" not in r
        assert r["total_symbols"] == 0

        destroy_emulator(session_id=sid)

    def test_load_binary_tool(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator,
            load_binary, read_memory,
        )

        r = create_emulator(arch="x86_32")
        sid = r["session_id"]

        # Load hex-encoded "mov eax, 42" at 0x1000
        code_hex = "b82a000000"
        r = load_binary(session_id=sid, data=code_hex, address=0x1000, entry_point=0x1000)
        assert "error" not in r
        assert r["size"] == 5
        assert r["entry_point"] == 0x1000

        # Verify data was written.
        r = read_memory(session_id=sid, address=0x1000, size=5)
        assert r["data"] == code_hex

        destroy_emulator(session_id=sid)


# ---------------------------------------------------------------------------
# Bug fixes and improvements (new tests)
# ---------------------------------------------------------------------------

class TestWritePermissions:
    """Bug 1: write_memory must respect region permissions."""

    def test_write_readonly_region_raises(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000, parse_perms("r"))
        with pytest.raises(ValueError, match="not writable"):
            session.write_memory(0x1000, b"\x90")

    def test_write_readexec_region_raises(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000, parse_perms("rx"))
        with pytest.raises(ValueError, match="not writable"):
            session.write_memory(0x1000, b"\x90")

    def test_write_writable_region_succeeds(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000, parse_perms("rw"))
        written = session.write_memory(0x1000, b"\x90\xc3")
        assert written == 2
        assert session.read_memory(0x1000, 2) == b"\x90\xc3"

    def test_write_rwx_region_succeeds(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)  # default is rwx
        written = session.write_memory(0x1000, b"\xAA")
        assert written == 1


    def test_write_spanning_two_adjacent_regions_succeeds(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000, parse_perms("rw"))
        session.map_memory(0x2000, 0x1000, parse_perms("rw"))
        # Write spanning boundary between two adjacent writable regions.
        written = session.write_memory(0x1FF0, b"\xAA" * 32)
        assert written == 32
        assert session.read_memory(0x1FF0, 32) == b"\xAA" * 32

    def test_write_into_unmapped_space_raises(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000, parse_perms("rw"))
        # Write starts in mapped region but extends past it into unmapped space.
        with pytest.raises(ValueError, match="unmapped gap"):
            session.write_memory(0x1FF0, b"\xAA" * 32)

    def test_write_completely_unmapped_raises(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000, parse_perms("rw"))
        with pytest.raises(ValueError, match="no mapped region"):
            session.write_memory(0x5000, b"\x90")

class TestTimeoutDetection:
    """Bug 2: emulate should report timeout and elapsed_ms."""

    def test_timeout_stop_reason(self, manager: SessionManager) -> None:
        """Infinite loop with short timeout should report 'timeout'."""
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        # Infinite loop: jmp to self
        encoding, _ = ks.asm("label: jmp label", addr=0x1000)
        session.write_memory(0x1000, bytes(encoding))

        # Timeout of 50ms = 50000 us
        result = session.emulate(
            address=0x1000, stop_address=0xFFFF, timeout_us=50_000
        )
        assert result["stop_reason"] == "timeout"

    def test_elapsed_ms_present(self, manager: SessionManager) -> None:
        """Every emulate result should include elapsed_ms."""
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 42")
        session.write_memory(0x1000, bytes(encoding))

        result = session.emulate(address=0x1000, count=1)
        assert "elapsed_ms" in result
        assert isinstance(result["elapsed_ms"], float)
        assert result["elapsed_ms"] >= 0


class TestReadWatchpointValue:
    """Bug 3: read watchpoint value must reflect actual memory content."""

    def test_read_watchpoint_value(self, manager: SessionManager) -> None:
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)  # code
        session.map_memory(0x2000, 0x1000)  # data

        # Write a known value at 0x2000.
        session.write_memory(0x2000, b"\x42\x00\x00\x00")

        ks = Ks(arch.ks_arch, arch.ks_mode)
        # mov eax, dword ptr [0x2000]  — reads from 0x2000
        encoding, _ = ks.asm("mov eax, dword ptr [0x2000]")
        session.write_memory(0x1000, bytes(encoding))

        session.add_watchpoint(0x2000, size=4, access="r")
        result = session.emulate(address=0x1000, count=10)
        assert result["stop_reason"] == "watchpoint"
        assert result["watchpoint"]["access"] == "read"
        # The actual value at 0x2000 is 0x42 (little-endian).
        assert result["watchpoint"]["value"] == 0x42


class TestHexdumpCollapse:
    """Improvement 1: hexdump collapses repeated lines with '*'."""

    def test_hexdump_collapse_zeros(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        # 64 bytes of zeros: first line + '*' + no trailing distinct line
        dump = session.hexdump(0x1000, 64)
        lines = dump.strip().split("\n")
        assert len(lines) == 2
        assert lines[1] == "*"

    def test_hexdump_no_collapse_distinct(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        # Write distinct data in each 16-byte line
        for i in range(4):
            session.write_memory(0x1000 + i * 16, bytes([i + 1] * 16))
        dump = session.hexdump(0x1000, 64)
        lines = dump.strip().split("\n")
        # 4 distinct lines, no collapse
        assert len(lines) == 4
        assert "*" not in dump


class TestTraceSymbols:
    """Improvement 3: trace entries and step output include symbol names."""

    def test_trace_includes_symbols(self, manager: SessionManager) -> None:
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2")
        session.write_memory(0x1000, bytes(encoding))

        session.add_symbol("entry", 0x1000)
        session.enable_trace()
        session.emulate(address=0x1000, count=2)
        trace = session.get_trace()
        assert trace["entries"][0].get("symbol") == "entry"
        # Second instruction has no symbol.
        assert "symbol" not in trace["entries"][1]

    def test_step_includes_symbol(self, manager: SessionManager) -> None:
        from keystone import Ks

        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 42")
        session.write_memory(0x1000, bytes(encoding))

        session.add_symbol("main", 0x1000)
        result = session.step(address=0x1000)
        assert result.get("symbol") == "main"


class TestServerBugFixes:
    """Server-level tests for bug fixes and improvements."""

    def test_write_memory_readonly_returns_error(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory, write_memory,
        )

        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000, perms="r")
        r = write_memory(session_id=sid, address=0x1000, data="90")
        assert "error" in r
        assert "not writable" in r["error"]
        destroy_emulator(session_id=sid)

    def test_error_format_no_double_quotes(self) -> None:
        """Fix 4: KeyError messages should not have extra quotes."""
        from mcp_emulate.server import destroy_emulator

        r = destroy_emulator(session_id="bogus-session-id")
        assert "error" in r
        # Should NOT start with a quote character (the repr wrapping).
        assert not r["error"].startswith("'"), f"Error has extra quotes: {r['error']!r}"
        assert not r["error"].startswith('"'), f"Error has extra quotes: {r['error']!r}"
        assert "No session with id" in r["error"]

    def test_hexdump_clamped_field(self) -> None:
        """Improvement 2: hexdump returns clamped flag when size exceeds 4096."""
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory, hexdump,
        )

        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x2000)

        # Request within limit — no clamped field.
        r = hexdump(session_id=sid, address=0x1000, size=256)
        assert "clamped" not in r
        assert r["size"] == 256

        # Request beyond limit — clamped field present.
        r = hexdump(session_id=sid, address=0x1000, size=8192)
        assert r["clamped"] is True
        assert r["requested_size"] == 8192
        assert r["size"] == 4096

        destroy_emulator(session_id=sid)

    def test_map_memory_rounded_up_from(self) -> None:
        """Improvement 5: map_memory returns rounded_up_from when size was rounded."""
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory,
        )

        r = create_emulator(arch="x86_32")
        sid = r["session_id"]

        # Size already page-aligned — no rounded_up_from.
        r = map_memory(session_id=sid, address=0x1000, size=0x1000)
        assert "rounded_up_from" not in r
        assert r["size"] == 0x1000

        # Size NOT page-aligned — rounded_up_from present.
        r = map_memory(session_id=sid, address=0x2000, size=100)
        assert r["rounded_up_from"] == 100
        assert r["size"] == 0x1000

        destroy_emulator(session_id=sid)

    def test_emulate_elapsed_ms_in_server(self) -> None:
        """Improvement 6: emulate response includes elapsed_ms."""
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory,
            write_memory, assemble, emulate,
        )

        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)

        asm = assemble(arch="x86_32", code="mov eax, 1", address=0x1000)
        write_memory(session_id=sid, address=0x1000, data=asm["bytes_hex"])

        r = emulate(session_id=sid, address=0x1000, count=1)
        assert "elapsed_ms" in r
        assert isinstance(r["elapsed_ms"], float)

        destroy_emulator(session_id=sid)


# ---------------------------------------------------------------------------
# Feature 7: MIPS and RISC-V Architectures
# ---------------------------------------------------------------------------

class TestNewArchitectures:
    """Feature 7: MIPS32, MIPS32BE, RISC-V 32/64."""

    @pytest.mark.parametrize("arch_name", ["mips32", "mips32be", "riscv32", "riscv64"])
    def test_new_arch_has_required_fields(self, arch_name: str) -> None:
        cfg = get_arch(arch_name)
        assert cfg.name == arch_name
        assert cfg.pc_reg in cfg.register_map
        assert cfg.sp_reg in cfg.register_map

    @pytest.mark.parametrize("arch_name", ["mips32", "mips32be", "riscv32", "riscv64"])
    def test_new_arch_session_creation(self, manager: SessionManager, arch_name: str) -> None:
        session = manager.create(arch_name)
        assert session.arch.name == arch_name
        # Can read/write registers.
        regs = session.get_registers()
        assert "pc" in regs
        assert "sp" in regs

    @pytest.mark.parametrize("arch_name", ["mips32", "mips32be", "riscv32", "riscv64"])
    def test_new_arch_memory_map_and_rw(self, manager: SessionManager, arch_name: str) -> None:
        session = manager.create(arch_name)
        session.map_memory(0x10000, 0x1000)
        session.write_memory(0x10000, b"\xde\xad\xbe\xef")
        data = session.read_memory(0x10000, 4)
        assert data == b"\xde\xad\xbe\xef"

    def test_mips32_assemble_and_run(self, manager: SessionManager) -> None:
        """MIPS32 has full Keystone support — assemble and execute."""
        from keystone import Ks
        arch = get_arch("mips32")
        session = manager.create("mips32")
        session.map_memory(0x10000, 0x1000)

        # addiu $v0, $zero, 42  -> sets v0 to 42
        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("addiu $v0, $zero, 42")
        code = bytes(encoding)
        session.write_memory(0x10000, code)
        result = session.emulate(address=0x10000, count=1)
        assert result["stop_reason"] == "count_exhausted"
        assert result["registers"]["v0"] == 42

    def test_riscv_no_assemble(self) -> None:
        """RISC-V has no Keystone backend — assemble tool returns error."""
        from mcp_emulate.server import assemble
        r = assemble(arch="riscv32", code="addi x1, x0, 1")
        assert "error" in r
        assert "not supported" in r["error"].lower() or "no Keystone" in r["error"]

    def test_riscv_disassemble(self) -> None:
        """RISC-V disassembly works via Capstone."""
        from mcp_emulate.server import disassemble
        # addi x1, x0, 1 = 0x00100093 in RISC-V 32 little-endian
        r = disassemble(arch="riscv32", data="93001000")
        assert "error" not in r
        assert len(r["instructions"]) >= 1


# ---------------------------------------------------------------------------
# Feature 1: Memory Snapshots and Diff
# ---------------------------------------------------------------------------

class TestMemorySnapshots:
    def test_snapshot_and_diff_no_changes(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        session.write_memory(0x1000, b"\x41\x42\x43\x44")

        session.snapshot_memory("before")
        session.snapshot_memory("after")
        diff = session.diff_memory("before", "after")
        assert diff["change_count"] == 0
        assert diff["changes"] == []

    def test_snapshot_and_diff_with_changes(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        session.write_memory(0x1000, b"\x41\x42\x43\x44")
        session.snapshot_memory("before")

        session.write_memory(0x1000, b"\xFF\x42\x43\xEE")
        session.snapshot_memory("after")

        diff = session.diff_memory("before", "after")
        assert diff["change_count"] >= 1
        # Byte 0 changed from 0x41->0xFF, byte 3 changed from 0x44->0xEE.
        change_addrs = [c["address"] for c in diff["changes"]]
        assert 0x1000 in change_addrs  # first byte changed

    def test_snapshot_missing_label(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.snapshot_memory("a")
        with pytest.raises(KeyError, match="No memory snapshot"):
            session.diff_memory("a", "nonexistent")

    def test_snapshot_overwrites(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        r1 = session.snapshot_memory("snap")
        assert r1["label"] == "snap"
        r2 = session.snapshot_memory("snap")  # overwrite
        assert r2["label"] == "snap"

    def test_snapshot_tool(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory,
            write_memory, snapshot_memory, diff_memory,
        )
        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)
        write_memory(session_id=sid, address=0x1000, data="41424344")
        r = snapshot_memory(session_id=sid, label="before")
        assert "error" not in r
        assert r["label"] == "before"

        write_memory(session_id=sid, address=0x1000, data="ff4243ee")
        snapshot_memory(session_id=sid, label="after")

        r = diff_memory(session_id=sid, label_a="before", label_b="after")
        assert "error" not in r
        assert r["change_count"] >= 1
        destroy_emulator(session_id=sid)


# ---------------------------------------------------------------------------
# Feature 2: Stack View
# ---------------------------------------------------------------------------

class TestStackView:
    def test_get_stack_basic(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x7F000, 0x1000)
        # Set SP to top of mapped region.
        session.set_register("esp", 0x7F000)
        # Write some known values.
        import struct
        for i in range(4):
            session.write_memory(0x7F000 + i * 4, struct.pack("<I", 0xDEAD0000 + i))
        result = session.get_stack(count=4)
        assert result["sp"] == 0x7F000
        assert result["pointer_size"] == 4
        assert result["count"] == 4
        assert result["entries"][0]["value"] == 0xDEAD0000
        assert result["entries"][1]["value"] == 0xDEAD0001

    def test_get_stack_with_symbols(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x7F000, 0x1000)
        session.set_register("esp", 0x7F000)
        import struct
        session.write_memory(0x7F000, struct.pack("<I", 0x401000))
        session.add_symbol("main", 0x401000)
        result = session.get_stack(count=1)
        assert result["entries"][0]["symbol"] == "main"

    def test_get_stack_tool(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory,
            write_memory, set_registers, get_stack,
        )
        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x7F000, size=0x1000)
        set_registers(session_id=sid, values={"esp": 0x7F000})
        write_memory(session_id=sid, address=0x7F000, data="efbeadde")
        r = get_stack(session_id=sid, count=1)
        assert "error" not in r
        assert r["count"] >= 1
        assert r["entries"][0]["value"] == 0xDEADBEEF
        destroy_emulator(session_id=sid)


# ---------------------------------------------------------------------------
# Feature 5: Memory Map Visualization
# ---------------------------------------------------------------------------

class TestMemoryMap:
    def test_memory_map_basic(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)  # rwx by default
        text = session.memory_map()
        assert "0x00001000" in text
        assert "rwx" in text

    def test_memory_map_gap_shown(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        session.map_memory(0x5000, 0x1000)
        text = session.memory_map()
        assert "gap" in text

    def test_memory_map_symbols(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        session.add_symbol("entry", 0x1000)
        text = session.memory_map()
        assert "entry" in text

    def test_memory_map_tool(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory, memory_map,
        )
        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)
        r = memory_map(session_id=sid)
        assert "error" not in r
        assert r["region_count"] == 1
        assert "0x00001000" in r["map"]
        destroy_emulator(session_id=sid)


# ---------------------------------------------------------------------------
# Feature 8: Trace Diff
# ---------------------------------------------------------------------------

class TestTraceDiff:
    def test_save_and_diff_identical_traces(self, manager: SessionManager) -> None:
        from keystone import Ks
        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2")
        code = bytes(encoding)
        session.write_memory(0x1000, code)

        session.enable_trace()
        session.emulate(address=0x1000, count=2)
        session.save_trace("run1")

        # Identical second run.
        session.enable_trace()
        session.emulate(address=0x1000, count=2)
        session.save_trace("run2")

        diff = session.diff_trace("run1", "run2")
        assert diff["common_prefix"] == 2
        assert diff["divergence_index"] is None
        assert diff["divergences"] == []

    def test_save_and_diff_divergent_traces(self, manager: SessionManager) -> None:
        from keystone import Ks
        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)

        # Run 1: mov eax, 1; mov ebx, 2
        enc1, _ = ks.asm("mov eax, 1; mov ebx, 2")
        session.write_memory(0x1000, bytes(enc1))
        session.enable_trace()
        session.emulate(address=0x1000, count=2)
        session.save_trace("run1")

        # Run 2: mov eax, 1; mov ecx, 3 (different second instruction)
        enc2, _ = ks.asm("mov eax, 1; mov ecx, 3")
        session.write_memory(0x1000, bytes(enc2))
        session.enable_trace()
        session.emulate(address=0x1000, count=2)
        session.save_trace("run2")

        diff = session.diff_trace("run1", "run2")
        assert diff["common_prefix"] == 1  # first instruction matches
        assert diff["divergence_index"] == 1
        assert len(diff["divergences"]) >= 1

    def test_diff_trace_missing_label(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        with pytest.raises(KeyError, match="No saved trace"):
            session.diff_trace("nope", "also_nope")

    def test_save_trace_tool(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory,
            write_memory, assemble, enable_trace, emulate,
            save_trace, diff_trace,
        )
        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)

        asm = assemble(arch="x86_32", code="mov eax, 1; mov ebx, 2", address=0x1000)
        write_memory(session_id=sid, address=0x1000, data=asm["bytes_hex"])

        enable_trace(session_id=sid)
        emulate(session_id=sid, address=0x1000, count=2)
        r = save_trace(session_id=sid, label="run1")
        assert "error" not in r
        assert r["entries"] == 2

        enable_trace(session_id=sid)
        emulate(session_id=sid, address=0x1000, count=2)
        save_trace(session_id=sid, label="run2")

        r = diff_trace(session_id=sid, label_a="run1", label_b="run2")
        assert "error" not in r
        assert r["common_prefix"] == 2
        destroy_emulator(session_id=sid)


# ---------------------------------------------------------------------------
# Feature 6: Conditional Breakpoints
# ---------------------------------------------------------------------------

class TestConditionalBreakpoints:
    def test_conditional_breakpoint_fires_when_met(self, manager: SessionManager) -> None:
        """Breakpoint with 'eax == 1' fires after mov eax,1."""
        from keystone import Ks
        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        # mov eax, 1; mov ebx, 2  (bp on second instruction)
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2")
        code = bytes(encoding)
        session.write_memory(0x1000, code)

        # Address of second instruction.
        bp_addr = 0x1000 + 5  # first mov is 5 bytes
        session.add_breakpoint(bp_addr, condition="eax == 1")
        result = session.emulate(address=0x1000, count=10)
        assert result["stop_reason"] == "breakpoint"
        assert result["breakpoint_address"] == bp_addr

    def test_conditional_breakpoint_skipped_when_not_met(self, manager: SessionManager) -> None:
        """Breakpoint with 'eax == 99' does NOT fire when eax is 1."""
        from keystone import Ks
        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2")
        code = bytes(encoding)
        session.write_memory(0x1000, code)

        bp_addr = 0x1000 + 5
        session.add_breakpoint(bp_addr, condition="eax == 99")
        result = session.emulate(address=0x1000, count=2)
        assert result["stop_reason"] == "count_exhausted"

    def test_unconditional_breakpoint_still_works(self, manager: SessionManager) -> None:
        """No condition = always fire (backward compat)."""
        from keystone import Ks
        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2")
        code = bytes(encoding)
        session.write_memory(0x1000, code)

        bp_addr = 0x1000 + 5
        session.add_breakpoint(bp_addr)  # no condition
        result = session.emulate(address=0x1000, count=10)
        assert result["stop_reason"] == "breakpoint"

    def test_invalid_condition_rejected(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        with pytest.raises(ValueError, match="Invalid condition"):
            session.add_breakpoint(0x1000, condition="garbage!!!")

    def test_invalid_register_in_condition_rejected(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        with pytest.raises(ValueError, match="Unknown register"):
            session.add_breakpoint(0x1000, condition="xyz == 1")

    def test_condition_with_and_or(self, manager: SessionManager) -> None:
        from keystone import Ks
        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        # mov eax, 1; mov ebx, 2; nop
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2; nop")
        session.write_memory(0x1000, bytes(encoding))

        # BP on nop (offset 10), condition: eax == 1 and ebx == 2
        bp_addr = 0x1000 + 10
        session.add_breakpoint(bp_addr, condition="eax == 1 and ebx == 2")
        result = session.emulate(address=0x1000, count=10)
        assert result["stop_reason"] == "breakpoint"

    def test_conditional_breakpoint_tool(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory,
            write_memory, assemble, add_breakpoint,
            list_breakpoints, emulate,
        )
        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)

        asm = assemble(arch="x86_32", code="mov eax, 1; mov ebx, 2", address=0x1000)
        write_memory(session_id=sid, address=0x1000, data=asm["bytes_hex"])

        r = add_breakpoint(session_id=sid, address=0x1005, condition="eax == 1")
        assert "error" not in r
        assert r["condition"] == "eax == 1"

        r = list_breakpoints(session_id=sid)
        assert r["breakpoints"][0]["condition"] == "eax == 1"

        r = emulate(session_id=sid, address=0x1000, count=10)
        assert r["stop_reason"] == "breakpoint"
        destroy_emulator(session_id=sid)


# ---------------------------------------------------------------------------
# Feature 4: Syscall Hooking
# ---------------------------------------------------------------------------

class TestSyscallHooking:
    def test_hook_syscall_x86_64_skip(self, manager: SessionManager) -> None:
        """x86_64 syscall instruction gets hooked in skip mode."""
        from keystone import Ks
        arch = get_arch("x86_64")
        session = manager.create("x86_64")
        session.map_memory(0x1000, 0x1000)

        # mov rax, 1 (write syscall); syscall; mov rbx, 1
        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov rax, 1; syscall; mov rbx, 1", addr=0x1000)
        session.write_memory(0x1000, bytes(encoding))

        session.hook_syscall(mode="skip", default_return=42)
        result = session.emulate(address=0x1000, count=3)
        # Should have completed all 3 instructions.
        assert result["stop_reason"] == "count_exhausted"
        # rax should be 42 (return from syscall), rbx should be 1.
        assert result["registers"]["rax"] == 42
        assert result["registers"]["rbx"] == 1

        log = session.get_syscall_log()
        assert log["total"] == 1
        assert log["entries"][0]["syscall_nr"] == 1

    def test_hook_syscall_stop_mode(self, manager: SessionManager) -> None:
        from keystone import Ks
        arch = get_arch("x86_64")
        session = manager.create("x86_64")
        session.map_memory(0x1000, 0x1000)

        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov rax, 60; syscall; mov rbx, 1", addr=0x1000)
        session.write_memory(0x1000, bytes(encoding))

        session.hook_syscall(mode="stop")
        result = session.emulate(address=0x1000, count=10)
        # Should have stopped at syscall, NOT executed mov rbx.
        assert result["registers"]["rbx"] == 0
        log = session.get_syscall_log()
        assert log["total"] == 1
        assert log["entries"][0]["syscall_nr"] == 60

    def test_unhook_syscall(self, manager: SessionManager) -> None:
        session = manager.create("x86_64")
        session.hook_syscall(mode="skip")
        result = session.unhook_syscall()
        assert result["unhooked"] is True

    def test_hook_invalid_mode(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        with pytest.raises(ValueError, match="Invalid syscall mode"):
            session.hook_syscall(mode="invalid")

    def test_hook_syscall_tool(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory,
            write_memory, assemble, hook_syscall,
            unhook_syscall, get_syscall_log, emulate,
        )
        r = create_emulator(arch="x86_64")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)

        asm = assemble(arch="x86_64", code="mov rax, 1; syscall", address=0x1000)
        write_memory(session_id=sid, address=0x1000, data=asm["bytes_hex"])

        r = hook_syscall(session_id=sid, mode="skip", default_return=0)
        assert "error" not in r

        emulate(session_id=sid, address=0x1000, count=2)

        r = get_syscall_log(session_id=sid)
        assert "error" not in r
        assert r["total"] == 1

        r = unhook_syscall(session_id=sid)
        assert "error" not in r
        destroy_emulator(session_id=sid)


# ---------------------------------------------------------------------------
# Feature 3: Executable Loader (basic — no real binary, just tests error path)
# ---------------------------------------------------------------------------

class TestExecutableLoader:
    def test_load_invalid_binary_raises(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        with pytest.raises(ValueError, match="Failed to parse"):
            session.load_executable(b"not a real binary")

    def test_load_executable_tool_invalid(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, load_executable,
        )
        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        r = load_executable(session_id=sid, data="deadbeef")
        assert "error" in r
        destroy_emulator(session_id=sid)


# ---------------------------------------------------------------------------
# Feature 9: Session Serialization
# ---------------------------------------------------------------------------

class TestSessionSerialization:
    def test_export_import_roundtrip(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        session.write_memory(0x1000, b"\xde\xad\xbe\xef")
        session.set_register("eax", 42)
        session.add_breakpoint(0x1000)
        session.add_symbol("main", 0x1000)

        state = session.export_state()
        assert state["version"] == 1
        assert state["arch"] == "x86_32"
        assert len(state["regions"]) == 1
        assert state["registers"]["eax"] == 42
        assert len(state["breakpoints"]) == 1
        assert len(state["symbols"]) == 1

        # Import into new session.
        session2 = manager.create("x86_32")
        session2.import_state(state)
        assert session2.read_memory(0x1000, 4) == b"\xde\xad\xbe\xef"
        assert session2.get_register("eax") == 42
        assert len(session2.breakpoints) == 1
        assert len(session2.list_symbols()) == 1

    def test_import_arch_mismatch(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        state = session.export_state()
        session2 = manager.create("x86_64")
        with pytest.raises(ValueError, match="does not match"):
            session2.import_state(state)

    def test_export_import_tools(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory,
            write_memory, set_registers, add_symbol,
            export_session, import_session,
        )
        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)
        write_memory(session_id=sid, address=0x1000, data="deadbeef")
        set_registers(session_id=sid, values={"eax": 99})
        add_symbol(session_id=sid, name="start", address=0x1000)

        state = export_session(session_id=sid)
        assert "error" not in state
        assert state["version"] == 1

        r = import_session(arch="x86_32", state=state)
        assert "error" not in r
        assert "session_id" in r
        assert r["regions_restored"] == 1
        assert r["symbols_restored"] == 1

        destroy_emulator(session_id=sid)
        destroy_emulator(session_id=r["session_id"])


# ---------------------------------------------------------------------------
# Bug-fix regression tests
# ---------------------------------------------------------------------------


class TestBugFixes:
    """Tests for bug fixes."""

    def test_load_binary_into_rx_region(self, manager: SessionManager) -> None:
        """BUG-1: load_binary should bypass permission check."""
        session = manager.create("x86_32")
        # Map as rx (no write permission).
        from unicorn import UC_PROT_READ, UC_PROT_EXEC
        session.map_memory(0x1000, 0x1000, UC_PROT_READ | UC_PROT_EXEC)
        # load_binary should succeed — it uses uc.mem_write() directly.
        result = session.load_binary(b"\x90\x90\x90", 0x1000)
        assert result["size"] == 3
        assert session.read_memory(0x1000, 3) == b"\x90\x90\x90"

    def test_search_memory_address_without_size(self, manager: SessionManager) -> None:
        """BUG-7: search_memory should raise when address given without size."""
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        with pytest.raises(ValueError, match="size is required"):
            session.search_memory(b"\x00", address=0x1000)

    def test_watchpoint_big_endian(self, manager: SessionManager) -> None:
        """BUG-2: watchpoint should report correct value for big-endian."""
        from keystone import Ks
        arch = get_arch("mips32be")
        session = manager.create("mips32be")
        session.map_memory(0x1000, 0x1000)
        # lui loads upper 16 bits, so lui $t1, 2 gives $t1 = 0x20000.
        # Map data region at 0x20000.
        session.map_memory(0x20000, 0x1000)
        # Write big-endian 0x00000042 at 0x20000.
        session.uc.mem_write(0x20000, b"\x00\x00\x00\x42")
        session.add_watchpoint(0x20000, size=4, access="r")
        ks = Ks(arch.ks_arch, arch.ks_mode)
        # lui $t1, 2 => $t1 = 0x20000; lw $t0, 0($t1) => read [0x20000]
        code = "lui $t1, 2; lw $t0, 0($t1)"
        encoding, _ = ks.asm(code, addr=0x1000)
        session.uc.mem_write(0x1000, bytes(encoding))
        result = session.emulate(address=0x1000, count=10)
        assert result["stop_reason"] == "watchpoint"
        # Value should be 0x42 (read as big-endian).
        assert result["watchpoint"]["value"] == 0x42

    def test_step_at_breakpoint(self, manager: SessionManager) -> None:
        """BUG-5: step() at a breakpoint address should not get stuck."""
        from keystone import Ks
        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2")
        session.write_memory(0x1000, bytes(encoding))
        # Set breakpoint at first instruction.
        session.add_breakpoint(0x1000)
        # Step should execute the instruction and not report breakpoint.
        result = session.step(0x1000)
        assert result["stop_reason"] != "breakpoint"
        assert result["address"] == 0x1000
        # Breakpoint should still be in place.
        assert 0x1000 in session.breakpoints

    def test_diff_memory_size_change(self, manager: SessionManager) -> None:
        """BUG-3: diff_memory should report size differences."""
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        session.snapshot_memory("before")
        # Unmap and remap with different size.
        session.unmap_memory(0x1000, 0x1000)
        session.map_memory(0x1000, 0x2000)
        session.snapshot_memory("after")
        diff = session.diff_memory("before", "after")
        # The old 0x1000 region (4096 bytes) is gone, replaced by 0x1000 region (8192 bytes).
        # Since the address is the same but size changed, we should see size_changed.
        size_changes = [c for c in diff["changes"] if c.get("type") == "size_changed"]
        assert len(size_changes) == 1
        assert size_changes[0]["old_size"] == 0x1000
        assert size_changes[0]["new_size"] == 0x2000

    def test_trace_overflow_tracking(self, manager: SessionManager) -> None:
        """BUG-8: trace should track total instructions even when deque overflows."""
        from keystone import Ks
        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2; mov ecx, 3; mov edx, 4; inc eax")
        session.write_memory(0x1000, bytes(encoding))
        session.enable_trace(max_entries=2)
        session.emulate(address=0x1000, count=5)
        trace = session.get_trace()
        assert trace["available"] == 2
        assert trace["total_traced"] == 5
        # Indices should be absolute: 3 and 4 (last 2 of 5).
        assert trace["entries"][0]["index"] == 3
        assert trace["entries"][1]["index"] == 4


# ---------------------------------------------------------------------------
# Fault hooks
# ---------------------------------------------------------------------------


class TestFaultHooks:
    """Tests for structured fault reporting."""

    def test_memory_read_unmapped(self, manager: SessionManager) -> None:
        from keystone import Ks
        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        ks = Ks(arch.ks_arch, arch.ks_mode)
        # Read from unmapped address 0xDEAD0000.
        encoding, _ = ks.asm("mov eax, [0xDEAD0000]")
        session.write_memory(0x1000, bytes(encoding))
        result = session.emulate(address=0x1000, count=1)
        assert "fault" in result
        assert result["fault"]["type"] == "memory"
        assert result["fault"]["access"] == "read_unmapped"
        assert result["fault"]["address"] == 0xDEAD0000

    def test_memory_write_unmapped(self, manager: SessionManager) -> None:
        from keystone import Ks
        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov dword ptr [0xDEAD0000], eax")
        session.write_memory(0x1000, bytes(encoding))
        result = session.emulate(address=0x1000, count=1)
        assert "fault" in result
        assert result["fault"]["type"] == "memory"
        assert result["fault"]["access"] == "write_unmapped"

    def test_no_fault_on_normal_execution(self, manager: SessionManager) -> None:
        from keystone import Ks
        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 42")
        session.write_memory(0x1000, bytes(encoding))
        result = session.emulate(address=0x1000, count=1)
        assert "fault" not in result


# ---------------------------------------------------------------------------
# Unmap memory
# ---------------------------------------------------------------------------


class TestUnmapMemory:
    def test_unmap_basic(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        session.write_memory(0x1000, b"\xde\xad")
        result = session.unmap_memory(0x1000, 0x1000)
        assert result["address"] == 0x1000
        assert result["size"] == 0x1000
        assert len(session.mapped_regions) == 0
        # Reading should now fail.
        with pytest.raises(Exception):
            session.read_memory(0x1000, 2)

    def test_unmap_tool(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory, unmap_memory,
        )
        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)
        r = unmap_memory(session_id=sid, address=0x1000, size=0x1000)
        assert "error" not in r
        assert r["size"] == 0x1000
        destroy_emulator(session_id=sid)

    def test_unmap_removes_tracking(self, manager: SessionManager) -> None:
        """After unmap, mapped_regions must not contain the unmapped region."""
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        assert len(session.mapped_regions) == 1
        session.unmap_memory(0x1000, 0x1000)
        assert len(session.mapped_regions) == 0
        # Double-check: list_regions should be empty.
        assert session.list_regions() == []


# ---------------------------------------------------------------------------
# Protect memory
# ---------------------------------------------------------------------------


class TestProtectMemory:
    def test_protect_removes_write(self, manager: SessionManager) -> None:
        from unicorn import UC_PROT_READ, UC_PROT_EXEC
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)  # rwx by default
        session.write_memory(0x1000, b"\xaa\xbb")
        session.protect_memory(0x1000, 0x1000, UC_PROT_READ | UC_PROT_EXEC)
        # Read should still work.
        assert session.read_memory(0x1000, 2) == b"\xaa\xbb"
        # Write (via write_memory with perm check) should fail.
        with pytest.raises(ValueError):
            session.write_memory(0x1000, b"\xcc")

    def test_protect_tool(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory, protect_memory,
        )
        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)
        r = protect_memory(session_id=sid, address=0x1000, size=0x1000, perms="rx")
        assert "error" not in r
        assert r["perms"] == 5  # UC_PROT_READ | UC_PROT_EXEC
        destroy_emulator(session_id=sid)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_coverage_basic(self, manager: SessionManager) -> None:
        from keystone import Ks
        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2; mov ecx, 3")
        session.write_memory(0x1000, bytes(encoding))
        session.enable_coverage()
        session.emulate(address=0x1000, count=3)
        cov = session.get_coverage()
        assert cov["total_blocks_hit"] >= 1
        assert len(cov["blocks"]) >= 1
        # First block should be at 0x1000.
        assert cov["blocks"][0]["address"] == 0x1000
        assert cov["blocks"][0]["count"] >= 1

    def test_coverage_disable(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        result = session.enable_coverage()
        assert result["enabled"] is True
        result = session.disable_coverage()
        assert result["enabled"] is False

    def test_coverage_tools(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory, write_memory,
            assemble, emulate, enable_coverage, disable_coverage, get_coverage,
        )
        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)
        asm = assemble(arch="x86_32", code="mov eax, 1; mov ebx, 2", address=0x1000)
        write_memory(session_id=sid, address=0x1000, data=asm["bytes_hex"])
        r = enable_coverage(session_id=sid)
        assert "error" not in r
        emulate(session_id=sid, address=0x1000, count=2)
        r = get_coverage(session_id=sid)
        assert "error" not in r
        assert r["total_blocks_hit"] >= 1
        r = disable_coverage(session_id=sid)
        assert "error" not in r
        assert r["enabled"] is False
        destroy_emulator(session_id=sid)


    def test_coverage_pagination(self, manager: SessionManager) -> None:
        from keystone import Ks
        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        ks = Ks(arch.ks_arch, arch.ks_mode)
        encoding, _ = ks.asm("mov eax, 1; mov ebx, 2; mov ecx, 3")
        session.write_memory(0x1000, bytes(encoding))
        session.enable_coverage()
        session.emulate(address=0x1000, count=3)
        # Get all coverage.
        full = session.get_coverage(offset=0, limit=1000)
        total = full["total_blocks_hit"]
        assert total >= 1
        # Paginate with limit=1.
        page = session.get_coverage(offset=0, limit=1)
        assert len(page["blocks"]) <= 1
        assert page["total_blocks_hit"] == total
        assert page["offset"] == 0
        assert page["limit"] == 1

# ---------------------------------------------------------------------------
# Convenience tools (session level)
# ---------------------------------------------------------------------------


class TestConvenienceTools:
    def test_setup_stack(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        result = session.setup_stack(0x7FFF0000, 0x10000)
        assert result["address"] == 0x7FFF0000
        assert result["size"] == 0x10000
        assert result["sp"] == 0x7FFF0000 + 0x10000
        # SP register should be set.
        assert session.get_register("esp") == result["sp"]

    def test_assemble_and_load(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        result = session.assemble_and_load("mov eax, 42; inc eax", 0x1000)
        assert result["address"] == 0x1000
        assert result["entry_point"] == 0x1000
        assert result["statement_count"] == 2
        # PC should be set to 0x1000.
        assert session.get_register("eip") == 0x1000

    def test_assemble_and_load_no_keystone(self, manager: SessionManager) -> None:
        session = manager.create("riscv32")
        with pytest.raises(ValueError, match="no Keystone backend"):
            session.assemble_and_load("nop", 0x1000)

    def test_diff_context(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        session.set_register("eax", 10)
        session.save_context("before")
        session.set_register("eax", 20)
        session.set_register("ebx", 99)
        session.save_context("after")
        diff = session.diff_context("before", "after")
        assert diff["changed_count"] >= 2
        assert "eax" in diff["changes"]
        assert diff["changes"]["eax"]["old"] == 10
        assert diff["changes"]["eax"]["new"] == 20
        assert "ebx" in diff["changes"]

    def test_fill_memory(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        written = session.fill_memory(0x1000, 8, b"\xab\xcd")
        assert written == 8
        assert session.read_memory(0x1000, 8) == b"\xab\xcd" * 4

    def test_fill_memory_empty_pattern(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        with pytest.raises(ValueError, match="Pattern must not be empty"):
            session.fill_memory(0x1000, 8, b"")

    def test_nop_out(self, manager: SessionManager) -> None:
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        session.write_memory(0x1000, b"\xcc" * 4)
        result = session.nop_out(0x1000, 4)
        assert result["nop_count"] == 4
        assert session.read_memory(0x1000, 4) == b"\x90" * 4

    def test_nop_out_arm(self, manager: SessionManager) -> None:
        session = manager.create("arm")
        session.map_memory(0x1000, 0x1000)
        result = session.nop_out(0x1000, 8)
        assert result["nop_count"] == 2
        nop = b"\x00\xf0\x20\xe3"
        assert session.read_memory(0x1000, 8) == nop * 2

    def test_nop_out_bad_size(self, manager: SessionManager) -> None:
        session = manager.create("arm")
        session.map_memory(0x1000, 0x1000)
        with pytest.raises(ValueError, match="not a multiple"):
            session.nop_out(0x1000, 3)

    def test_run_and_diff(self, manager: SessionManager) -> None:
        from keystone import Ks
        arch = get_arch("x86_32")
        session = manager.create("x86_32")
        session.map_memory(0x1000, 0x1000)
        session.map_memory(0x2000, 0x1000)
        ks = Ks(arch.ks_arch, arch.ks_mode)
        # mov eax, 42; mov dword ptr [0x2000], eax
        encoding, _ = ks.asm("mov eax, 42; mov dword ptr [0x2000], eax")
        session.write_memory(0x1000, bytes(encoding))
        result = session.run_and_diff(address=0x1000, count=2)
        assert result["stop_reason"] == "count_exhausted"
        assert "register_diff" in result
        assert "eax" in result["register_diff"]
        assert result["register_diff"]["eax"]["new"] == 42
        assert "memory_diff" in result
        assert result["registers_changed"] >= 1


# ---------------------------------------------------------------------------
# Convenience server tools
# ---------------------------------------------------------------------------


class TestConvenienceServerTools:
    def test_setup_stack_tool(self) -> None:
        from mcp_emulate.server import create_emulator, destroy_emulator, setup_stack
        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        r = setup_stack(session_id=sid)
        assert "error" not in r
        assert r["sp"] > r["address"]
        destroy_emulator(session_id=sid)

    def test_assemble_and_load_tool(self) -> None:
        from mcp_emulate.server import create_emulator, destroy_emulator, assemble_and_load
        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        r = assemble_and_load(session_id=sid, code="mov eax, 1; mov ebx, 2", address=0x1000)
        assert "error" not in r
        assert r["statement_count"] == 2
        destroy_emulator(session_id=sid)

    def test_diff_context_tool(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, set_registers,
            save_context, diff_context,
        )
        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        set_registers(session_id=sid, values={"eax": 1})
        save_context(session_id=sid, label="a")
        set_registers(session_id=sid, values={"eax": 99})
        save_context(session_id=sid, label="b")
        r = diff_context(session_id=sid, label_a="a", label_b="b")
        assert "error" not in r
        assert r["changed_count"] >= 1
        assert "eax" in r["changes"]
        destroy_emulator(session_id=sid)

    def test_fill_memory_tool(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory, fill_memory, read_memory,
        )
        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)
        r = fill_memory(session_id=sid, address=0x1000, size=8, pattern="abcd")
        assert "error" not in r
        r = read_memory(session_id=sid, address=0x1000, size=8)
        assert r["data"] == "abcdabcdabcdabcd"
        destroy_emulator(session_id=sid)

    def test_nop_out_tool(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory, nop_out, read_memory,
        )
        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)
        r = nop_out(session_id=sid, address=0x1000, size=4)
        assert "error" not in r
        assert r["nop_count"] == 4
        r = read_memory(session_id=sid, address=0x1000, size=4)
        assert r["data"] == "90909090"
        destroy_emulator(session_id=sid)

    def test_run_and_diff_tool(self) -> None:
        from mcp_emulate.server import (
            create_emulator, destroy_emulator, map_memory,
            write_memory, assemble, run_and_diff,
        )
        r = create_emulator(arch="x86_32")
        sid = r["session_id"]
        map_memory(session_id=sid, address=0x1000, size=0x1000)
        asm = assemble(arch="x86_32", code="mov eax, 42", address=0x1000)
        write_memory(session_id=sid, address=0x1000, data=asm["bytes_hex"])
        r = run_and_diff(session_id=sid, address=0x1000, count=1)
        assert "error" not in r
        assert "register_diff" in r
        assert "memory_diff" in r
        destroy_emulator(session_id=sid)


# ---------------------------------------------------------------------------
# ArchConfig fields
# ---------------------------------------------------------------------------


class TestArchConfigFields:
    """Tests for new ArchConfig fields."""

    def test_endian_field(self) -> None:
        assert get_arch("x86_32").endian == "little"
        assert get_arch("mips32be").endian == "big"
        assert get_arch("arm64").endian == "little"

    def test_nop_bytes_field(self) -> None:
        assert get_arch("x86_32").nop_bytes == b"\x90"
        assert get_arch("arm").nop_bytes == b"\x00\xf0\x20\xe3"
        assert get_arch("riscv32").nop_bytes == b"\x13\x00\x00\x00"

    def test_all_archs_have_nop_bytes(self) -> None:
        for name, arch in ARCHITECTURES.items():
            assert len(arch.nop_bytes) > 0, f"{name} has empty nop_bytes"
            assert len(arch.nop_bytes) in (1, 4), f"{name} has unexpected NOP size {len(arch.nop_bytes)}"


# ---------------------------------------------------------------------------
# Detect arch
# ---------------------------------------------------------------------------


class TestDetectArch:
    def test_detect_elf_x86_32(self) -> None:
        from mcp_emulate.server import detect_arch
        import struct
        # Minimal ELF32 header (52 bytes).
        elf = bytearray(52)
        elf[0:4] = b"\x7fELF"      # e_ident magic
        elf[4] = 1                   # EI_CLASS: ELFCLASS32
        elf[5] = 1                   # EI_DATA: ELFDATA2LSB
        elf[6] = 1                   # EI_VERSION
        struct.pack_into("<H", elf, 18, 3)  # e_machine: EM_386 = 3
        r = detect_arch(data=elf.hex())
        assert "error" not in r
        assert r["arch"] == "x86_32"
        assert r["format"] == "elf"
        assert r["endian"] == "little"

    def test_detect_invalid_binary(self) -> None:
        from mcp_emulate.server import detect_arch
        r = detect_arch(data="00")
        assert "error" in r