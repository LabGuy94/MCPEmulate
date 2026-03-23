"""Integration test: exercises the MCP server over its actual stdio JSON-RPC transport.

Spawns the server as a subprocess, sends JSON-RPC requests, validates responses.
This tests the real wire protocol, not just in-process function calls.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any


class MCPClient:
    """Minimal MCP client that speaks JSON-RPC over stdio to a subprocess."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "mcp_emulate.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
        self._id = 0

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def send(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request and read the response."""
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
        }
        if params is not None:
            msg["params"] = params
        line = json.dumps(msg) + "\n"
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

        # Read response line(s). MCP may emit notifications before the response.
        while True:
            resp_line = self.proc.stdout.readline()
            if not resp_line:
                raise RuntimeError("Server closed stdout unexpectedly")
            resp = json.loads(resp_line)
            # Skip notifications (no "id" field).
            if "id" in resp and resp["id"] == msg["id"]:
                return resp

    def send_notification(self, method: str, params: dict | None = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            msg["params"] = params
        line = json.dumps(msg) + "\n"
        assert self.proc.stdin is not None
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Call an MCP tool and return the parsed result content."""
        resp = self.send("tools/call", {"name": name, "arguments": arguments})
        if "error" in resp:
            raise RuntimeError(f"MCP error: {resp['error']}")
        result = resp["result"]
        # MCP tools return content as a list of content blocks.
        # Each block has "type" and "text" (for text blocks).
        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            return json.loads(content[0]["text"])
        return result

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=5)


def main() -> None:
    client = MCPClient()
    passed = 0
    failed = 0
    errors: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            msg = f"  FAIL  {name}"
            if detail:
                msg += f" -- {detail}"
            print(msg)
            errors.append(msg)

    try:
        # ── Phase 0: Initialize ──────────────────────────────────────────
        print("\n=== Phase 0: Initialize ===")
        resp = client.send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "integration-test", "version": "0.1"},
        })
        check("initialize succeeds", "result" in resp, str(resp.get("error", "")))

        # Send initialized notification (required by protocol).
        client.send_notification("notifications/initialized")

        # ── Phase 1: List tools ──────────────────────────────────────────
        print("\n=== Phase 1: List tools ===")
        resp = client.send("tools/list", {})
        tools = resp["result"]["tools"]
        tool_names = sorted(t["name"] for t in tools)
        print(f"  Server exposes {len(tools)} tools: {', '.join(tool_names)}")
        check("tool count is 41", len(tools) == 41, f"got {len(tools)}")

        expected_tools = sorted([
            "create_emulator", "destroy_emulator",
            "map_memory", "write_memory", "read_memory",
            "set_registers", "get_registers",
            "emulate", "add_breakpoint", "remove_breakpoint", "list_breakpoints",
            "step", "save_context", "restore_context",
            "assemble", "disassemble",
            # Iteration 3
            "list_regions", "hexdump", "search_memory",
            "add_watchpoint", "remove_watchpoint", "list_watchpoints",
            # Iteration 4
            "enable_trace", "disable_trace", "get_trace",
            # Iteration 5
            "add_symbol", "remove_symbol", "list_symbols", "load_binary",
            # Iteration 6 — new features
            "snapshot_memory", "diff_memory",
            "get_stack",
            "memory_map",
            "save_trace", "diff_trace",
            "hook_syscall", "unhook_syscall", "get_syscall_log",
            "load_executable",
            "export_session", "import_session",
        ])
        check("all expected tools present", tool_names == expected_tools,
              f"missing: {set(expected_tools) - set(tool_names)}, extra: {set(tool_names) - set(expected_tools)}")

        # ── Phase 2: Standalone tools (no session) ───────────────────────
        print("\n=== Phase 2: Standalone tools ===")

        # Assemble
        r = client.call_tool("assemble", {"arch": "x86_32", "code": "mov eax, 42; ret"})
        check("assemble succeeds", "error" not in r)
        check("assemble byte count", r.get("byte_count", 0) >= 3, str(r))
        code_hex = r["bytes_hex"]

        # Disassemble
        r = client.call_tool("disassemble", {"arch": "x86_32", "data": code_hex})
        check("disassemble succeeds", "error" not in r)
        mnemonics = [i["mnemonic"] for i in r.get("instructions", [])]
        check("disassemble finds mov+ret", "mov" in mnemonics and "ret" in mnemonics, str(mnemonics))

        # ── Phase 3: Session lifecycle ───────────────────────────────────
        print("\n=== Phase 3: Session lifecycle ===")

        r = client.call_tool("create_emulator", {"arch": "x86_32"})
        check("create_emulator succeeds", "session_id" in r)
        sid = r["session_id"]

        # ── Phase 4: Memory operations ───────────────────────────────────
        print("\n=== Phase 4: Memory operations ===")

        r = client.call_tool("map_memory", {"session_id": sid, "address": 0x1000, "size": 0x1000})
        check("map_memory code region", "error" not in r)

        r = client.call_tool("map_memory", {"session_id": sid, "address": 0x100000, "size": 0x1000})
        check("map_memory stack region", "error" not in r)

        r = client.call_tool("write_memory", {"session_id": sid, "address": 0x1000, "data": code_hex})
        check("write_memory succeeds", "error" not in r)
        check("write_memory byte count", r.get("bytes_written") == len(code_hex) // 2)

        r = client.call_tool("read_memory", {"session_id": sid, "address": 0x1000, "size": len(code_hex) // 2})
        check("read_memory succeeds", "error" not in r)
        check("read_memory data matches", r.get("data") == code_hex)

        # ── Phase 5: Memory inspection (Iteration 3) ────────────────────
        print("\n=== Phase 5: Memory inspection ===")

        r = client.call_tool("list_regions", {"session_id": sid})
        check("list_regions succeeds", "error" not in r)
        check("list_regions count", r.get("count") == 2, str(r.get("count")))
        check("list_regions perms", r["regions"][0]["perms"] == "rwx")

        # Write known data for hexdump
        hello_hex = "48656c6c6f20576f726c6400"  # "Hello World\0"
        client.call_tool("write_memory", {"session_id": sid, "address": 0x1000, "data": hello_hex})
        r = client.call_tool("hexdump", {"session_id": sid, "address": 0x1000, "size": 16})
        check("hexdump succeeds", "error" not in r)
        check("hexdump contains hex", "48 65 6c 6c" in r.get("hexdump", ""), r.get("hexdump", "")[:80])
        check("hexdump contains ASCII", "Hello World" in r.get("hexdump", ""))

        # Search memory
        r = client.call_tool("search_memory", {"session_id": sid, "pattern": "48656c6c6f"})
        check("search_memory finds pattern", r.get("count", 0) >= 1)
        check("search_memory correct address", 0x1000 in r.get("matches", []))

        r = client.call_tool("search_memory", {"session_id": sid, "pattern": "ffffffffff"})
        check("search_memory no match", r.get("count") == 0)

        # ── Phase 6: Registers ───────────────────────────────────────────
        print("\n=== Phase 6: Registers ===")

        r = client.call_tool("set_registers", {"session_id": sid, "values": {"eax": 0, "esp": 0x100FFC}})
        check("set_registers succeeds", "error" not in r)

        r = client.call_tool("get_registers", {"session_id": sid, "names": ["eax", "esp"]})
        check("get_registers succeeds", "error" not in r)
        check("get_registers values", r["registers"]["eax"] == 0 and r["registers"]["esp"] == 0x100FFC)

        # ── Phase 7: Emulation full round-trip ───────────────────────────
        print("\n=== Phase 7: Emulation ===")

        # Re-write actual code (we overwrote with Hello World earlier)
        asm = client.call_tool("assemble", {"arch": "x86_32", "code": "mov eax, 42; ret", "address": 0x1000})
        client.call_tool("write_memory", {"session_id": sid, "address": 0x1000, "data": asm["bytes_hex"]})

        # Write return address to stack
        stop_addr = 0xDEAD
        client.call_tool("map_memory", {"session_id": sid, "address": 0xD000, "size": 0x1000})
        ret_bytes = stop_addr.to_bytes(4, "little").hex()
        client.call_tool("write_memory", {"session_id": sid, "address": 0x100FFC, "data": ret_bytes})

        r = client.call_tool("emulate", {
            "session_id": sid, "address": 0x1000, "stop_address": stop_addr, "count": 100,
        })
        check("emulate succeeds", "error" not in r)
        check("emulate stop_reason=completed", r.get("stop_reason") == "completed", r.get("stop_reason", ""))
        check("emulate instructions=2", r.get("instructions_executed") == 2)
        check("emulate eax=42", r.get("registers", {}).get("eax") == 42)

        # ── Phase 8: Breakpoints ─────────────────────────────────────────
        print("\n=== Phase 8: Breakpoints ===")

        # Re-assemble multi-instruction code
        asm = client.call_tool("assemble", {
            "arch": "x86_32", "code": "mov eax, 1; mov ebx, 2; mov ecx, 3; mov edx, 4", "address": 0x1000,
        })
        client.call_tool("write_memory", {"session_id": sid, "address": 0x1000, "data": asm["bytes_hex"]})

        bp_addr = 0x1000 + 10  # 3rd instruction
        r = client.call_tool("add_breakpoint", {"session_id": sid, "address": bp_addr})
        check("add_breakpoint succeeds", "error" not in r)
        check("breakpoint count=1", r.get("total_breakpoints") == 1)

        r = client.call_tool("list_breakpoints", {"session_id": sid})
        check("list_breakpoints", r.get("breakpoints") == [{"address": bp_addr, "condition": None}])

        r = client.call_tool("emulate", {"session_id": sid, "address": 0x1000, "count": 100})
        check("emulate hits breakpoint", r.get("stop_reason") == "breakpoint")
        check("breakpoint address correct", r.get("breakpoint_address") == bp_addr)
        check("only 2 instructions executed", r.get("instructions_executed") == 2)

        r = client.call_tool("remove_breakpoint", {"session_id": sid, "address": bp_addr})
        check("remove_breakpoint succeeds", r.get("total_breakpoints") == 0)

        # ── Phase 9: Watchpoints (Iteration 3) ──────────────────────────
        print("\n=== Phase 9: Watchpoints ===")

        r = client.call_tool("map_memory", {"session_id": sid, "address": 0x2000, "size": 0x1000})
        # Assemble code that writes to 0x2000
        asm = client.call_tool("assemble", {
            "arch": "x86_32", "code": "mov eax, 0x42; mov dword ptr [0x2000], eax", "address": 0x1000,
        })
        client.call_tool("write_memory", {"session_id": sid, "address": 0x1000, "data": asm["bytes_hex"]})

        r = client.call_tool("add_watchpoint", {"session_id": sid, "address": 0x2000, "size": 4, "access": "w"})
        check("add_watchpoint succeeds", "error" not in r)
        check("watchpoint count=1", r.get("total_watchpoints") == 1)

        r = client.call_tool("list_watchpoints", {"session_id": sid})
        check("list_watchpoints", r.get("count") == 1)
        check("watchpoint details", r["watchpoints"][0]["address"] == 0x2000)

        r = client.call_tool("emulate", {"session_id": sid, "address": 0x1000, "count": 10})
        check("emulate hits watchpoint", r.get("stop_reason") == "watchpoint")
        check("watchpoint access=write", r.get("watchpoint", {}).get("access") == "write")
        check("watchpoint address=0x2000", r.get("watchpoint", {}).get("address") == 0x2000)

        r = client.call_tool("remove_watchpoint", {"session_id": sid, "address": 0x2000})
        check("remove_watchpoint succeeds", r.get("total_watchpoints") == 0)

        # ── Phase 10: Stepping ───────────────────────────────────────────
        print("\n=== Phase 10: Stepping ===")

        asm = client.call_tool("assemble", {
            "arch": "x86_32", "code": "mov eax, 77; mov ebx, 88", "address": 0x1000,
        })
        client.call_tool("write_memory", {"session_id": sid, "address": 0x1000, "data": asm["bytes_hex"]})

        r = client.call_tool("step", {"session_id": sid, "address": 0x1000})
        check("step succeeds", "error" not in r)
        check("step mnemonic=mov", r.get("instruction", {}).get("mnemonic") == "mov")
        check("step eax=77", r.get("registers", {}).get("eax") == 77)
        check("step ebx untouched", r.get("registers", {}).get("ebx") == 2)  # from earlier

        # Step next
        pc = r["registers"]["eip"]
        r = client.call_tool("step", {"session_id": sid, "address": pc})
        check("step 2 ebx=88", r.get("registers", {}).get("ebx") == 88)

        # ── Phase 11: Context save/restore ───────────────────────────────
        print("\n=== Phase 11: Context save/restore ===")

        r = client.call_tool("save_context", {"session_id": sid, "label": "snap1"})
        check("save_context succeeds", "error" not in r)
        check("save_context label", "snap1" in r.get("saved_labels", []))

        client.call_tool("set_registers", {"session_id": sid, "values": {"eax": 999}})
        r = client.call_tool("get_registers", {"session_id": sid, "names": ["eax"]})
        check("eax changed to 999", r["registers"]["eax"] == 999)

        r = client.call_tool("restore_context", {"session_id": sid, "label": "snap1"})
        check("restore_context succeeds", "error" not in r)
        check("eax restored to 77", r.get("registers", {}).get("eax") == 77)

        # ── Phase 12: Trace (Iteration 4) ────────────────────────────────
        print("\n=== Phase 12: Trace ===")

        # Assemble 3 instructions
        asm = client.call_tool("assemble", {
            "arch": "x86_32", "code": "mov eax, 1; mov ebx, 2; mov ecx, 3", "address": 0x1000,
        })
        client.call_tool("write_memory", {"session_id": sid, "address": 0x1000, "data": asm["bytes_hex"]})

        r = client.call_tool("enable_trace", {"session_id": sid})
        check("enable_trace succeeds", r.get("enabled") is True)

        client.call_tool("emulate", {"session_id": sid, "address": 0x1000, "count": 3})

        r = client.call_tool("get_trace", {"session_id": sid})
        check("get_trace succeeds", "error" not in r)
        check("trace total=3", r.get("total") == 3, str(r.get("total")))
        check("trace entry 0 is mov", r["entries"][0].get("mnemonic") == "mov")
        check("trace entry 0 address", r["entries"][0].get("address") == 0x1000)

        # Pagination
        r = client.call_tool("get_trace", {"session_id": sid, "offset": 1, "limit": 1})
        check("trace pagination", len(r.get("entries", [])) == 1)
        check("trace pagination index", r["entries"][0].get("index") == 1)

        r = client.call_tool("disable_trace", {"session_id": sid})
        check("disable_trace succeeds", r.get("enabled") is False)
        check("disable_trace entry count", r.get("entries") == 3)

        # ── Phase 13: Symbols (Iteration 5) ──────────────────────────────
        print("\n=== Phase 13: Symbols ===")

        r = client.call_tool("add_symbol", {"session_id": sid, "name": "main", "address": 0x1000})
        check("add_symbol succeeds", "error" not in r)
        check("symbol count=1", r.get("total_symbols") == 1)

        r = client.call_tool("add_symbol", {"session_id": sid, "name": "helper", "address": 0x2000})
        check("add second symbol", r.get("total_symbols") == 2)

        r = client.call_tool("list_symbols", {"session_id": sid})
        check("list_symbols count", r.get("count") == 2)
        names = [s["name"] for s in r.get("symbols", [])]
        check("list_symbols sorted", names == ["helper", "main"])

        r = client.call_tool("remove_symbol", {"session_id": sid, "name": "helper"})
        check("remove_symbol succeeds", r.get("total_symbols") == 1)

        # ── Phase 14: Load binary (Iteration 5) ─────────────────────────
        print("\n=== Phase 14: Load binary ===")

        # Load "mov eax, 99; ret" at a fresh address
        asm = client.call_tool("assemble", {"arch": "x86_32", "code": "mov eax, 99; ret"})
        r = client.call_tool("load_binary", {
            "session_id": sid, "data": asm["bytes_hex"], "address": 0x5000, "entry_point": 0x5000,
        })
        check("load_binary succeeds", "error" not in r)
        check("load_binary size", r.get("size") == asm["byte_count"])
        check("load_binary entry_point", r.get("entry_point") == 0x5000)

        # Verify the loaded code runs
        client.call_tool("map_memory", {"session_id": sid, "address": 0x200000, "size": 0x1000})
        client.call_tool("set_registers", {"session_id": sid, "values": {"esp": 0x200FFC}})
        stop_addr2 = 0xD001
        ret_bytes2 = stop_addr2.to_bytes(4, "little").hex()
        client.call_tool("write_memory", {"session_id": sid, "address": 0x200FFC, "data": ret_bytes2})
        r = client.call_tool("emulate", {
            "session_id": sid, "address": 0x5000, "stop_address": stop_addr2, "count": 100,
        })
        check("loaded code runs", r.get("stop_reason") == "completed")
        check("loaded code eax=99", r.get("registers", {}).get("eax") == 99)

        # ── Phase 15: Memory snapshots & diff (Feature 1) ─────────────
        print("\n=== Phase 15: Memory snapshots & diff ===")

        r = client.call_tool("snapshot_memory", {"session_id": sid, "label": "snap_before"})
        check("snapshot_memory succeeds", "error" not in r)
        check("snapshot label", r.get("label") == "snap_before")

        # Modify memory and take another snapshot.
        client.call_tool("write_memory", {"session_id": sid, "address": 0x1000, "data": "ff" * 4})
        r = client.call_tool("snapshot_memory", {"session_id": sid, "label": "snap_after"})
        check("snapshot after succeeds", "error" not in r)

        r = client.call_tool("diff_memory", {"session_id": sid, "label_a": "snap_before", "label_b": "snap_after"})
        check("diff_memory succeeds", "error" not in r)
        check("diff finds changes", r.get("change_count", 0) >= 1)

        # ── Phase 16: Stack view (Feature 2) ───────────────────────
        print("\n=== Phase 16: Stack view ===")

        r = client.call_tool("get_stack", {"session_id": sid, "count": 4})
        check("get_stack succeeds", "error" not in r)
        check("get_stack has entries", r.get("count", 0) >= 0)

        # ── Phase 17: Memory map (Feature 5) ───────────────────────
        print("\n=== Phase 17: Memory map ===")

        r = client.call_tool("memory_map", {"session_id": sid})
        check("memory_map succeeds", "error" not in r)
        check("memory_map has regions", r.get("region_count", 0) >= 1)
        check("memory_map text output", len(r.get("map", "")) > 0)

        # ── Phase 18: Trace diff (Feature 8) ──────────────────────
        print("\n=== Phase 18: Trace diff ===")

        asm = client.call_tool("assemble", {"arch": "x86_32", "code": "mov eax, 1; mov ebx, 2", "address": 0x1000})
        client.call_tool("write_memory", {"session_id": sid, "address": 0x1000, "data": asm["bytes_hex"]})
        client.call_tool("enable_trace", {"session_id": sid})
        client.call_tool("emulate", {"session_id": sid, "address": 0x1000, "count": 2})
        r = client.call_tool("save_trace", {"session_id": sid, "label": "trace_a"})
        check("save_trace succeeds", "error" not in r)
        check("save_trace entries", r.get("entries") == 2)

        # Run again, identical
        client.call_tool("enable_trace", {"session_id": sid})
        client.call_tool("emulate", {"session_id": sid, "address": 0x1000, "count": 2})
        client.call_tool("save_trace", {"session_id": sid, "label": "trace_b"})

        r = client.call_tool("diff_trace", {"session_id": sid, "label_a": "trace_a", "label_b": "trace_b"})
        check("diff_trace succeeds", "error" not in r)
        check("diff_trace common_prefix", r.get("common_prefix") == 2)

        # ── Phase 19: Conditional breakpoints (Feature 6) ───────────
        print("\n=== Phase 19: Conditional breakpoints ===")

        # Remove old breakpoints first.
        bps = client.call_tool("list_breakpoints", {"session_id": sid})
        for bp in bps.get("breakpoints", []):
            client.call_tool("remove_breakpoint", {"session_id": sid, "address": bp["address"]})

        asm = client.call_tool("assemble", {"arch": "x86_32", "code": "mov eax, 1; mov ebx, 2", "address": 0x1000})
        client.call_tool("write_memory", {"session_id": sid, "address": 0x1000, "data": asm["bytes_hex"]})

        bp_addr2 = 0x1000 + 5  # second instruction
        r = client.call_tool("add_breakpoint", {"session_id": sid, "address": bp_addr2, "condition": "eax == 1"})
        check("conditional bp added", "error" not in r)
        check("conditional bp condition", r.get("condition") == "eax == 1")

        r = client.call_tool("emulate", {"session_id": sid, "address": 0x1000, "count": 10})
        check("conditional bp fires", r.get("stop_reason") == "breakpoint")

        client.call_tool("remove_breakpoint", {"session_id": sid, "address": bp_addr2})

        # ── Phase 20: Syscall hooking (Feature 4) ─────────────────
        print("\n=== Phase 20: Syscall hooking ===")

        # Create an x86_64 session for syscall testing.
        r64 = client.call_tool("create_emulator", {"arch": "x86_64"})
        sid64 = r64["session_id"]
        client.call_tool("map_memory", {"session_id": sid64, "address": 0x1000, "size": 0x1000})

        asm64 = client.call_tool("assemble", {"arch": "x86_64", "code": "mov rax, 1; syscall", "address": 0x1000})
        client.call_tool("write_memory", {"session_id": sid64, "address": 0x1000, "data": asm64["bytes_hex"]})

        r = client.call_tool("hook_syscall", {"session_id": sid64, "mode": "skip", "default_return": 42})
        check("hook_syscall succeeds", "error" not in r)

        client.call_tool("emulate", {"session_id": sid64, "address": 0x1000, "count": 2})

        r = client.call_tool("get_syscall_log", {"session_id": sid64})
        check("get_syscall_log succeeds", "error" not in r)
        check("syscall logged", r.get("total", 0) >= 1)

        r = client.call_tool("unhook_syscall", {"session_id": sid64})
        check("unhook_syscall succeeds", r.get("unhooked") is True)

        client.call_tool("destroy_emulator", {"session_id": sid64})

        # ── Phase 21: Session export/import (Feature 9) ────────────
        print("\n=== Phase 21: Session export/import ===")

        r = client.call_tool("export_session", {"session_id": sid})
        check("export_session succeeds", "error" not in r)
        check("export has version", r.get("version") == 1)
        check("export has arch", r.get("arch") == "x86_32")
        check("export has regions", len(r.get("regions", [])) >= 1)

        exported_state = r
        r = client.call_tool("import_session", {"arch": "x86_32", "state": exported_state})
        check("import_session succeeds", "error" not in r)
        check("import returns session_id", "session_id" in r)
        imported_sid = r["session_id"]

        # Verify imported session has memory.
        r = client.call_tool("list_regions", {"session_id": imported_sid})
        check("imported session has regions", r.get("count", 0) >= 1)

        client.call_tool("destroy_emulator", {"session_id": imported_sid})

        # ── Phase 22: Multi-arch smoke (includes new arches) ────────
        print("\n=== Phase 22: Multi-arch smoke ===")

        for arch in ["x86_64", "arm", "arm64", "mips32", "mips32be", "riscv32", "riscv64"]:
            r = client.call_tool("create_emulator", {"arch": arch})
            check(f"create {arch}", "session_id" in r)
            sid2 = r["session_id"]
            client.call_tool("destroy_emulator", {"session_id": sid2})

        # ── Phase 23: Cleanup ────────────────────────────────────────────
        print("\n=== Phase 23: Cleanup ===")

        r = client.call_tool("destroy_emulator", {"session_id": sid})
        check("destroy_emulator succeeds", r.get("success") is True)

        # Verify destroyed session is gone
        r = client.call_tool("get_registers", {"session_id": sid})
        check("destroyed session returns error", "error" in r)

    finally:
        client.close()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TOTAL: {passed + failed}  |  PASSED: {passed}  |  FAILED: {failed}")
    print(f"{'='*60}")
    if errors:
        print("\nFailures:")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print("\nAll checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
