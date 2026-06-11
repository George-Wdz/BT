# Third-Party Components

This repository contains project-specific code and documentation. The following third-party repositories are used locally as reproduced research code or runtime dependencies, but are intentionally excluded from this GitHub repository.

| Local path | Upstream GitHub | Role in this project |
| --- | --- | --- |
| `Stage2/GPT4TS/` | https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All | Stage2 long-term time-series forecasting backend. |
| `MoE/llama-moe/` | https://github.com/pjlab-sys4nlp/llama-moe | Local MoE reference code and serving experiments. |

Notes:

- Keep upstream licenses, citations, and installation instructions.
- Do not upload third-party model weights, checkpoints, caches, or datasets to this repository.
- If these dependencies must be versioned later, use Git submodules or project forks instead of copying the full external repositories into this codebase.

