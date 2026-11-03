# APM + Prompt-CAM Repository

This repository contains two related fine-grained recognition pipelines built on top of a frozen DINOv2 backbone:

- `apm/` — the Attentive Prompt Mixer (APM) implementation
- `prompt_cam/` — the Prompt-CAM baseline implementation
- `shared/` — shared datasets and pretrained weights used by both methods

The project is organized by method, rather than by a single root-level training script.

## Repository layout

```text
APM_Final/
├── README.md
├── apm/
│   ├── main.py
│   ├── insertion_deletion.py
│   ├── make_heatmaps_cub.py
│   ├── logs_apm_dinov2_cub.txt
│   ├── logs_apm_dinov2_dog.txt
│   ├── logs_apm_dinov2_medleaf.txt
│   ├── logs_apm_dinov2_pet.txt
│   ├── checkpoints/
│   ├── data/
│   ├── engine/
│   ├── experiment/
│   │   └── config/
│   │       └── apm/
│   ├── heatmaps_apm_cub/
│   ├── insertion_deletion_out/
│   ├── model/
│   ├── output/
│   ├── pretrained_weights/
│   └── utils/
├── prompt_cam/
│   ├── main.py
│   ├── insertion_deletion_promptcam.py
│   ├── insertion_deletion_promptcam_cub.py
│   ├── make_heatmaps_cub.py
│   ├── logs_promptcam_dinov2_cub.txt
│   ├── logs_promptcam_dinov2_dog.txt
│   ├── logs_promptcam_dinov2_medleaf.txt
│   ├── logs_promptcam_dinov2_pet.txt
│   ├── checkpoints/
│   ├── data/
│   ├── engine/
│   ├── experiment/
│   │   └── config/
│   │       └── prompt_cam/
│   ├── heatmaps_promptcam_cub/
│   ├── insertion_deletion_out/
│   ├── model/
│   ├── output/
│   ├── pretrained_weights/
│   └── utils/
├── shared/
│   ├── data/
│   │   ├── annotations/
│   │   ├── dataset/
│   │   └── images/
│   │       ├── cub/
│   │       ├── dog/
│   │       ├── MedLeaf/
│   │       └── pet/
│   └── pretrained_weights/
└── .gitignore
```

## Setup

From the repo root:

```bash
cd /path/to/APM_Final
```

The project expects:

- PyTorch and torchvision
- DINOv2 ViT-B/14 weights in `shared/pretrained_weights/` or the configured checkpoint path
- dataset images under `shared/data/images/`

## Running APM

Train APM with a YAML config from the project root:

```bash
python apm/main.py --config apm/experiment/config/apm/dinov2/cub/args.yaml
```

Other dataset configs are available under:

```text
apm/experiment/config/apm/dinov2/
```

Examples:

```bash
python apm/main.py --config apm/experiment/config/apm/dinov2/dog/args.yaml
python apm/main.py --config apm/experiment/config/apm/dinov2/medleaf/args.yaml
python apm/main.py --config apm/experiment/config/apm/dinov2/pet/args.yaml
```

## Running Prompt-CAM

Train the baseline Prompt-CAM model:

```bash
python prompt_cam/main.py --config prompt_cam/experiment/config/prompt_cam/dinov2/cub/args.yaml
```

Other dataset configs:

```text
prompt_cam/experiment/config/prompt_cam/dinov2/
```

Examples:

```bash
python prompt_cam/main.py --config prompt_cam/experiment/config/prompt_cam/dinov2/dog/args.yaml
python prompt_cam/main.py --config prompt_cam/experiment/config/prompt_cam/dinov2/medleaf/args.yaml
python prompt_cam/main.py --config prompt_cam/experiment/config/prompt_cam/dinov2/pet/args.yaml
```

## Saliency / insertion-deletion evaluation

The repo includes dedicated evaluation scripts for the trained models.

For APM:

```bash
cd apm
python insertion_deletion.py --n_images 500
```

For Prompt-CAM:

```bash
cd prompt_cam
python insertion_deletion_promptcam.py --n_images 500
```

## Visualizations and outputs

Each method stores its generated artifacts in method-local directories:

- `apm/output/...` — training outputs and checkpoints
- `prompt_cam/output/...` — training outputs and checkpoints
- `apm/heatmaps_apm_cub/` — APM attention heatmaps
- `prompt_cam/heatmaps_promptcam_cub/` — Prompt-CAM heatmaps
- `apm/insertion_deletion_out/` — insertion/deletion metrics and plots
- `prompt_cam/insertion_deletion_out/` — insertion/deletion metrics and plots

## Notes

- There is no single root-level `scripts/` directory in the current repo state; training and evaluation are launched from the method folders (`apm/` and `prompt_cam/`).
- The shared datasets live in `shared/data/`, while method-specific checkpoints and outputs are stored inside each method folder.
- The YAML configs are the primary source of dataset and training settings for both methods.

## License

Add an appropriate license before public release.