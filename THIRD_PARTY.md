# Third-Party Components

This repository is intended to host project-specific code, documentation, and configuration for satellite-link rainfall retrieval and forecasting.

The following local directories are third-party or reproduced research code and are intentionally excluded from the main GitHub upload:

| Local path | Upstream/project role | How this project uses it |
| --- | --- | --- |
| `Stage2/GPT4TS/` | GPT4TS / One Fits All time-series forecasting code | Used as the Stage2 forecasting backend. Keep upstream attribution and citation from the original project. |
| `MoE/llama-moe/` | LLaMA-MoE research code | Used as a local reference/runtime for MoE and serving experiments. Keep upstream attribution and license terms. |

Large model weights, checkpoints, local caches, datasets, logs, and database snapshots are also excluded from GitHub. Put shareable datasets or model artifacts in a private Hugging Face Dataset/Model repository instead.

If these dependencies need to be versioned later, prefer one of:

1. Add the upstream repositories as Git submodules.
2. Fork the upstream repositories under the project organization and reference the fork.
3. Keep installation instructions in README files and do not vendor the full external code.

