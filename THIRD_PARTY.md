# Third-party model and redistribution boundary

This repository does **not** redistribute model weights.

The public aggregate evidence was produced with two openly licensed model families:

| Model | Public identifier | License reported by upstream |
| --- | --- | --- |
| DINOv2 ViT-S/14 | `dinov2_vits14` | Apache License 2.0 |
| SigLIP 2 base patch16 NaFlex | `google/siglip2-base-patch16-naflex` | Apache License 2.0 |

Upstream references:

- DINOv2 repository/model card: https://github.com/facebookresearch/dinov2
- SigLIP 2 model card: https://huggingface.co/google/siglip2-base-patch16-naflex

Only small reviewed aggregate measurements are versioned here. No upstream source code, checkpoints, safetensors, cached model snapshots, authentication tokens, or model-serving containers are included.

A separate private research workspace contained an additional model comparison with different distribution terms. That model and every access/reproduction artifact associated with it are intentionally absent from this public repository.

The procedural controlled scene and the publication code in this repository are project-authored materials. No license is granted for third-party model weights by this repository; users should consult the upstream model licenses before downloading or using those models themselves.
