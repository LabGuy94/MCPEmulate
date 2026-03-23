# MCPEmulate

> This project was vibecoded.

An [MCP](https://modelcontextprotocol.io/) server that exposes CPU emulation, disassembly, and assembly as tools for LLM agents. Built on [Unicorn](https://www.unicorn-engine.org/) (emulation), [Capstone](https://www.capstone-engine.org/) (disassembly), [Keystone](https://www.keystone-engine.org/) (assembly), and [LIEF](https://lief-project.github.io/) (binary parsing).

Agents can create isolated emulation sessions, load code or full executables, set breakpoints, hook syscalls, step through instructions, inspect memory and registers, and diff execution traces -- all through the standard MCP tool interface.

## Supported Architectures

| Architecture | Emulation | Disassembly | Assembly | Syscall Hooking |
|---|---|---|---|---|
| x86 (32-bit) | Yes | Yes | Yes | `int 0x80` |
| x86-64 | Yes | Yes | Yes | `syscall` |
| ARM (32-bit) | Yes | Yes | Yes | `svc 0` |
| AArch64 | Yes | Yes | Yes | `svc 0` |
| MIPS32 (LE) | Yes | Yes | Yes | `syscall` |
| MIPS32 (BE) | Yes | Yes | Yes | `syscall` |
| RISC-V 32 | Yes | Yes | No | `ecall` |
| RISC-V 64 | Yes | Yes | No | `ecall` |

RISC-V architectures lack a Keystone backend, so the `assemble` tool returns an error for them. Disassembly and emulation work normally.

## Install

Requires Python 3.10+.

```bash
# Run directly (no install needed)
uvx mcp-emulate

# Or install globally
uv pip install mcp-emulate
```

## Usage

### Claude Desktop / MCP Client

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "mcp-emulate": {
      "command": "uvx",
      "args": ["mcp-emulate"]
    }
  }
}
```

### CLI

```bash
# Default: stdio transport (for MCP clients)
mcp-emulate

# SSE transport (network, for web-based clients)
mcp-emulate --transport sse

# Streamable HTTP transport (newer MCP protocol)
mcp-emulate --transport streamable-http
```

## Tools (41)

### Session Management

| Tool | Description |
|---|---|
| `create_emulator` | Create a new emulation session for a given architecture |
| `destroy_emulator` | Destroy a session and free resources |
| `export_session` | Export full session state (memory, registers, breakpoints, symbols) to JSON |
| `import_session` | Create a new session and restore state from a previous export |

### Memory

| Tool | Description |
|---|---|
| `map_memory` | Map a memory region with specified permissions (r/w/x) |
| `write_memory` | Write hex or base64 data to memory |
| `read_memory` | Read memory as hex or base64 |
| `list_regions` | List all mapped regions |
| `hexdump` | Formatted hex dump (up to 4KB) with ASCII sidebar |
| `search_memory` | Search for byte patterns across mapped memory |
| `snapshot_memory` | Capture all memory content under a named label |
| `diff_memory` | Compare two snapshots and return changed byte ranges |
| `memory_map` | `/proc/self/maps`-style layout with gaps and symbol annotations |

### Registers

| Tool | Description |
|---|---|
| `set_registers` | Write one or more registers |
| `get_registers` | Read registers (specific or all) |
| `get_stack` | Read stack entries from SP, resolving values against symbols |

### Execution

| Tool | Description |
|---|---|
| `emulate` | Run emulation with stop address, instruction count, or timeout |
| `step` | Execute a single instruction with full disassembly |
| `add_breakpoint` | Set a breakpoint, optionally with a register condition |
| `remove_breakpoint` | Remove a breakpoint |
| `list_breakpoints` | List all breakpoints with their conditions |
| `save_context` | Save a register snapshot under a label |
| `restore_context` | Restore registers from a saved snapshot |

### Breakpoint Conditions

Conditional breakpoints accept expressions like:

```
eax == 42
rax > 0x1000 and rcx != 0
r0 == 0 or r1 & 0xff
```

Supported operators: `==`, `!=`, `>`, `<`, `>=`, `<=`, `&`. Connectives: `and`, `or`.

### Syscall Hooking

| Tool | Description |
|---|---|
| `hook_syscall` | Install a syscall hook (`skip` to log and continue, `stop` to halt) |
| `unhook_syscall` | Remove the syscall hook |
| `get_syscall_log` | Retrieve logged syscall invocations with pagination |

Each logged entry includes the syscall number, argument register values, and PC. The hook is architecture-aware -- it intercepts `int 0x80` on x86_32, `syscall` on x86_64, `svc 0` on ARM/AArch64, `syscall` on MIPS, and `ecall` on RISC-V.

### Watchpoints

| Tool | Description |
|---|---|
| `add_watchpoint` | Watch a memory address for read, write, or both |
| `remove_watchpoint` | Remove a watchpoint |
| `list_watchpoints` | List all active watchpoints |

### Tracing

| Tool | Description |
|---|---|
| `enable_trace` | Start recording executed instructions |
| `disable_trace` | Stop recording (log is preserved) |
| `get_trace` | Retrieve trace entries with disassembly and pagination |
| `save_trace` | Save the current trace log under a named label |
| `diff_trace` | Compare two saved traces instruction-by-instruction |

Trace diff reports the common prefix length, the divergence point, and up to 50 differing entries with full disassembly.

### Symbols

| Tool | Description |
|---|---|
| `add_symbol` | Associate a name with an address |
| `remove_symbol` | Remove a symbol |
| `list_symbols` | List all symbols |

Symbols are used to annotate stack entries, trace output, memory maps, and step results.

### Loading

| Tool | Description |
|---|---|
| `load_binary` | Load raw machine code at an address, auto-mapping memory |
| `load_executable` | Load an ELF, PE, or Mach-O binary with correct segment permissions, entry point, and symbols |
| `assemble` | Assemble instructions to machine code (standalone, no session) |
| `disassemble` | Disassemble machine code to instructions (standalone, no session) |

`load_executable` uses LIEF for format detection. It maps each loadable segment with the correct permissions, sets PC to the entry point, and registers exported symbols automatically.

## Example Workflow

A typical agent interaction:

1. `create_emulator(arch="x86_64")` -- start a session
2. `assemble(arch="x86_64", code="mov rax, 60; syscall")` -- assemble exit syscall
3. `load_binary(session_id=..., data=..., address=0x1000, entry_point=0x1000)` -- load code
4. `hook_syscall(session_id=..., mode="stop")` -- intercept syscalls
5. `enable_trace(session_id=...)` -- start recording
6. `emulate(session_id=..., address=0x1000, count=100)` -- run
7. `get_trace(session_id=...)` -- inspect what executed
8. `get_syscall_log(session_id=...)` -- see what syscalls were attempted
9. `export_session(session_id=...)` -- save state for later

## Architecture

```
src/mcp_emulate/
  architectures.py   Architecture configs, register maps, syscall conventions
  session.py         EmulationSession (Unicorn wrapper), SessionManager
  server.py          41 MCP tool handlers via FastMCP

tests/
  test_emulate.py              132 pytest unit tests
  test_server_integration.py   112 checks over JSON-RPC (23 phases)
```

### Key Design Decisions

- **`EmulationSession` uses `__slots__`** for memory efficiency and to catch typos. Every new attribute must be declared.
- **Breakpoints are `dict[int, str | None]`** (address to optional condition), not a set. This supports conditional breakpoints while keeping the same lookup semantics.
- **Syscall conventions are data, not code.** A frozen dataclass per architecture describes the hook type, interrupt number filter, register names for nr/args/return. The hooking logic is generic.
- **`ks_arch`/`ks_mode` are `Optional`** on `ArchConfig` so architectures without Keystone (RISC-V) can exist without a dummy value. The `assemble` tool checks this and returns a clear error.
- **`load_executable` writes via `uc.mem_write()` directly**, bypassing the permission check on `write_memory`. This is intentional -- binary loaders need to populate read-only segments.
- **Session serialization is versioned** (`"version": 1`) for forward compatibility.

## Development

```bash
git clone https://github.com/LabGuy94/MCPEmulate.git
cd MCPEmulate
uv venv .venv
uv pip install -e ".[dev]"
```

## Tests

```bash
# Unit tests (132 tests, ~1s)
uv run pytest tests/test_emulate.py -v

# Integration tests (112 checks over JSON-RPC subprocess, ~30s)
uv run python tests/test_server_integration.py

# Both
uv run pytest tests/ -v && uv run python tests/test_server_integration.py
```

## Dependencies

| Package | Purpose |
|---|---|
| [mcp](https://pypi.org/project/mcp/) >= 1.18.0 | MCP protocol / FastMCP server framework |
| [unicorn](https://pypi.org/project/unicorn/) >= 2.0.0 | CPU emulation engine |
| [capstone](https://pypi.org/project/capstone/) >= 5.0.0 | Disassembly engine |
| [keystone-engine](https://pypi.org/project/keystone-engine/) >= 0.9.2 | Assembly engine |
| [lief](https://pypi.org/project/lief/) >= 0.14.0 | ELF/PE/Mach-O binary parsing |

## License

GPL-2.0-only
