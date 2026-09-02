# Third-Party Components

This repository contains project-specific code and documentation. The following
third-party components are either vendored for the frontend or used locally as
research/runtime dependencies.

| Local path | Upstream | Role in this project |
| --- | --- | --- |
| `Stage1/multi_terminal_rain_demo/static/echarts.min.js` | https://echarts.apache.org/ | Vendored Apache ECharts 5.6.0 browser bundle (Apache-2.0). |
| `Stage2/GPT4TS/` | https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All | Stage2 long-term time-series forecasting backend. |
| `MoE/llama-moe/` | https://github.com/pjlab-sys4nlp/llama-moe | Local MoE reference code and serving experiments. |

Notes:

- Keep upstream licenses, citations, and installation instructions.
- Third-party model weights, checkpoints, caches, and datasets are kept outside this repository.
- If these dependencies must be versioned later, use Git submodules or project forks instead of copying the full external repositories into this codebase.
