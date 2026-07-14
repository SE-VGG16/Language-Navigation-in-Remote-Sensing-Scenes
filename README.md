# SOAR: Metric-Consistent Aerial Vision-Language Navigation

A modular PyTorch reference implementation of the telemetry-conditioned UAV vision-language navigation method described in:

> **Metric Consistent Aerial Perception for UAV Vision Language Navigation in Remote Sensing Scenes**

The repository separates the method into the following components:

1. **Telemetry-conditioned metric tokenizer** — rescales each frame to a fixed ground footprint and optionally rotates it to a north-up orientation.
2. **Spatial Vision Transformer encoder** — preserves the complete token grid rather than using a classification token.
3. **Near-orthographic BEV lift** — reshapes metric tokens into a bird's-eye-view feature map and optionally applies a supplied tilt-correction homography.
4. **Pose-registered BEV memory** — accumulates observations in an allocentric metric map using UAV planar position.
5. **Language grounding** — cross-attends from BEV cells to instruction tokens.
6. **Metric waypoint head** — predicts a spatial probability distribution and the expected waypoint in meters.

The repository also includes telemetry-error perturbation utilities, statistical bootstrap analysis, a synthetic smoke-test dataset, training and evaluation scripts, and unit tests.

## Important scope note

This is a **reference implementation reconstructed from the manuscript description**. The manuscript does not provide every benchmark-specific preprocessing choice, simulator interface, or complete set of hyperparameters. Therefore:

- the architecture is implemented faithfully at the component level;
- benchmark adapters for CityNav and AerialVLN-S must be connected to their official data/simulator interfaces;
- no unpublished experimental numbers are embedded in the code;
- the synthetic dataset is only a software smoke test and is not evidence of navigation performance;
- Scale-MAE or other pretrained weights must be supplied separately when used.

## Repository structure

```text
soar-uav-vln/
├── configs/
│   └── soar_tiny.yaml
├── examples/
│   └── demo.py
├── scripts/
│   ├── bootstrap_stats.py
│   ├── evaluate.py
│   ├── evaluate_telemetry.py
│   ├── make_synthetic_dataset.py
│   └── train.py
├── soar/
│   ├── bev_lift.py
│   ├── config.py
│   ├── data.py
│   ├── encoder.py
│   ├── geometry.py
│   ├── grounding.py
│   ├── losses.py
│   ├── memory.py
│   ├── metrics.py
│   ├── model.py
│   ├── telemetry_noise.py
│   ├── text.py
│   ├── tokenizer.py
│   └── waypoint.py
└── tests/
    └── test_model.py
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone <github-url>
cd soar-uav-vln
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

For the optional Hugging Face text encoder:

```bash
pip install -r requirements-full.txt
```

## Quick smoke test

```bash
python examples/demo.py
pytest -q
```

## Create a synthetic dataset

The generated data are not a scientific benchmark. They only verify the data pipeline.

```bash
python scripts/make_synthetic_dataset.py \
  --output data/synthetic \
  --num-episodes 128
```

## Train

```bash
python scripts/train.py \
  --config configs/soar_tiny.yaml \
  --data data/synthetic \
  --output runs/soar_tiny
```

If `--data` is omitted, the script uses an in-memory synthetic dataset.

## Evaluate

```bash
python scripts/evaluate.py \
  --config configs/soar_tiny.yaml \
  --checkpoint runs/soar_tiny/model_last.pt \
  --data data/synthetic \
  --output runs/soar_tiny/evaluation
```

Outputs:

- `per_episode.csv`
- `summary.json`

The included evaluator reports one-step waypoint error and success within a configurable radius. Full VLN metrics such as SR, OSR, SPL, and SDTW require rollout in the official benchmark environment.

## Telemetry robustness analysis

```bash
python scripts/evaluate_telemetry.py \
  --config configs/soar_tiny.yaml \
  --checkpoint runs/soar_tiny/model_last.pt \
  --data data/synthetic \
  --output runs/soar_tiny/telemetry_robustness.csv
```

The script evaluates controlled perturbations of:

- horizontal position;
- altitude;
- heading offset;
- heading random-walk drift;
- combined telemetry error.

## Paired bootstrap statistics

Evaluate two internally controlled models and then run:

```bash
python scripts/bootstrap_stats.py \
  --soar runs/soar/per_episode.csv \
  --control runs/control/per_episode.csv \
  --metrics waypoint_error_m success \
  --output runs/statistical_comparison.csv
```

The script aligns rows by `episode_id`, performs paired bootstrap resampling, reports 95% confidence intervals and two-sided p-values, and applies Holm correction.

## Expected episode format

A benchmark adapter should return a dictionary containing:

```python
{
    "images": Tensor[T, 3, H, W],          # float in [0, 1]
    "altitude_m": Tensor[T],
    "heading_deg": Tensor[T],              # clockwise from north
    "position_xy_m": Tensor[T, 2],         # x=east, y=north
    "instruction_ids": LongTensor[L],
    "instruction_mask": BoolTensor[L],     # True for valid tokens
    "target_waypoint_xy_m": Tensor[2],     # relative to final UAV pose
    "stop_target": Tensor[1],
    "episode_id": str,
}
```

All positions in one episode must use a common metric coordinate system. For long trajectories, use fixed-length windows and express the target waypoint relative to the final pose in each window.

## Coordinate conventions

- \(x\): east/right in meters.
- \(y\): north/up in meters.
- heading: clockwise from north, in degrees.
- image rows increase downward.
- the BEV map is north-up.
- waypoint coordinates are relative to the current UAV position.

## Pretrained visual encoder

The included encoder is a native PyTorch spatial ViT. A compatible checkpoint can be loaded with:

```python
model.encoder.load_checkpoint("path/to/checkpoint.pt")
```

Checkpoint conversion may be required if the source model uses a class token, a differently shaped position embedding, or different parameter names.

## Using a Hugging Face text encoder

Set in the YAML configuration:

```yaml
text:
  mode: hf
  model_name: sentence-transformers/all-MiniLM-L6-v2
  freeze: true
```

The model receives token IDs and attention masks prepared using the matching Hugging Face tokenizer. The default `tiny` mode uses the included deterministic hash tokenizer and is intended for testing or controlled training from scratch.

## Reproducibility recommendations

For paper experiments:

- run at least 3–5 independent seeds;
- store per-episode predictions;
- mark literature-only baselines separately from internally rerun controls;
- calculate paired confidence intervals only for models evaluated on identical episodes;
- report telemetry and terrain-stratified performance;
- log camera intrinsics, field of view, map resolution, window length, dataset version, hardware, and software versions.

## Limitations represented in the implementation

The current near-orthographic lift assumes a near-nadir view and a locally planar reference surface. A single homography cannot fully correct mountainous terrain, tall buildings, façades, or other depth-discontinuous scenes. For those settings, replace or augment the lift with elevation-aware, multi-plane, or learned depth-aware projection.

## License

MIT. See `LICENSE`.
