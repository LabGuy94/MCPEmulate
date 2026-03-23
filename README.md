# MCPEmulate

> This project was vibecoded.

MCP server that exposes CPU emulation (Unicorn), disassembly (Capstone), and assembly (Keystone) as tools for LLM clients.

## Setup

```bash
uv venv .venv
uv pip install -e ".[dev]"
```

## Usage

```json
{
  "mcpServers": {
    "mcp-emulate": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/MCPEmulate", "mcp-emulate"]
    }
  }
}
```

## Tests

```bash
uv run python -m pytest tests/ -v
```
