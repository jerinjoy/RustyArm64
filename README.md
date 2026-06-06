```
  ██████╗ ██╗   ██╗███████╗████████╗██╗   ██╗ █████╗ ██████╗ ███╗   ███╗ ██████╗ ██╗  ██╗
  ██╔══██╗██║   ██║██╔════╝╚══██╔══╝╚██╗ ██╔╝██╔══██╗██╔══██╗████╗ ████║██╔════╝ ██║  ██║
  ██████╔╝██║   ██║███████╗   ██║    ╚████╔╝ ███████║██████╔╝██╔████╔██║███████╗ ███████║
  ██╔══██╗██║   ██║╚════██║   ██║     ╚██╔╝  ██╔══██║██╔══██╗██║╚██╔╝██║██╔═══██╗╚════██║
  ██║  ██║╚██████╔╝███████║   ██║      ██║   ██║  ██║██║  ██║██║ ╚═╝ ██║╚██████╔╝     ██║
  ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝      ╚═╝
```

# RustyArm64

An ARM64 functional CPU simulator written in Rust, built incrementally by a
LangGraph orchestrator that loops an LLM through code → test → advance.

See [`docs/graph-overview.md`](docs/graph-overview.md) for the graph architecture.

## Usage

```bash
export DEEPSEEK_API_KEY="your_key"
cd orchestrator && uv run main.py
```
