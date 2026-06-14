# Detecting Defective Solar Panels with Computer Vision and Thermography

Published: draft  
Medium: [Detecting Defective Solar Panels with Computer Vision and Thermography](https://medium.com/@kyle-t-jones/detecting-defective-solar-panels-with-computer-vision-and-thermography-5cd0a43fc187)

## Business context

Solar energy is booming. Global solar capacity has grown from 40 GW in 2010 to over 1,000 GW in 2023. But here's the catch: solar panels fail. And when they do, they fail silently.

- Create dangerous hot spots (fire hazard)
- Reduce energy output by 10–40%
- Cost thousands in lost revenue annually
- Go undetected for months or years

Traditional manual inspection is slow, expensive, inconsistent, and dangerous. This repo explores computer vision and CLIP-based approaches for automated defect detection.

## Project layout

```
├── config.yaml              # Pipeline configuration
├── docs/
│   ├── article.md
│   ├── blog-33-solar-defect-detection.md
│   └── assets/
├── legacy/                  # Original notebook and monolithic script
├── scripts/
│   ├── run.py
│   └── solar_defect_visualizations.py
├── solar_panel_cv/
│   ├── cli.py               # CLI entry point
│   ├── config.py            # Pydantic config models
│   ├── context.py           # Shared pipeline state
│   ├── datasets.py          # TensorFlow dataset loaders
│   ├── clip.py              # CLIP predictor
│   ├── pipeline.py          # Step orchestration
│   └── steps/               # Individual pipeline steps
└── tests/
```

## Quick start

```bash
uv sync
uv run pytest

# List available steps
uv run solar-panel-cv --list-steps

# Run the full pipeline (requires Kaggle dataset download + GPU recommended)
uv run solar-panel-cv --config config.yaml

# Run a subset of steps
uv run solar-panel-cv --steps explore_kaggle_dataset clip_zero_shot_kaggle
```

Plots are saved to `outputs/` by default. Set `output.show_plots: true` in `config.yaml` to display them interactively.

## Configuration

All paths, hyperparameters, and step order live in `config.yaml`:

| Section | Controls |
|---------|----------|
| `data` | Kaggle dataset slug, archive extraction, binary dataset paths |
| `training` | Image size, batch size, epochs, learning rates |
| `model` | CLIP model ID |
| `clip` | Prompts, temperature search, threshold search |
| `output` | Plot directory and save/show behavior |
| `pipeline.steps` | Ordered list of steps to run |

Example — run only CLIP evaluation on an existing binary dataset:

```yaml
data:
  download_kaggle: false
pipeline:
  steps:
    - clip_binary_dataset_evaluation
    - clip_temperature_tuning
```

## Pipeline steps

| Step | Description |
|------|-------------|
| `explore_kaggle_dataset` | Sample images from the faulty panel dataset |
| `build_vgg16_and_explore` | Build VGG16 architecture and explore data |
| `train_multiclass_checkpoint` | Initial multiclass training |
| `fine_tune_multiclass` | Fine-tune VGG16 backbone |
| `clip_zero_shot_kaggle` | CLIP zero-shot on Kaggle data |
| `train_mobilenet_faulty_panel` | Train MobileNetV2 multiclass model |
| `train_multiclass_faulty_panel` | Train MobileNet on faulty panels (244px) |
| `extract_archive` | Extract local zip archive (optional) |
| `prepare_binary_dataset` | Build Clean / Not_Clean binary dataset |
| `compare_mobilenet_clip` | Side-by-side MobileNet vs CLIP |
| `train_binary_classifier` | Train binary MobileNet classifier |
| `clip_zero_shot_faulty_panel` | CLIP zero-shot on faulty panels |
| `clip_binary_evaluation` | CLIP binary eval with ROC analysis |
| `clip_binary_dataset_evaluation` | CLIP eval on prepared binary dataset |
| `clip_temperature_tuning` | Grid search temperature + threshold |
| `clip_weighted_confidence` | Class-weighted CLIP confidence scoring |

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).
