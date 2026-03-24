# NeRF / SIREN Integration

## Table of Contents
- [Overview](#overview)
- [Code Modifications](#code-modifications)
- [Installation](#installation)
- [Training](#training)
- [Inference & Rendering](#inference--rendering)
- [Checkpoints](#checkpoints)

---

## Overview

This project extends the original **NeRF** implementation by adding **SIREN** as an alternative model. You can easily switch between both models using a simple boolean flag.

---

## Code Modifications

### `src/run_platonerf.py`
- Modified the `create_nerf()` function  
- Supports instantiation of either **NeRF** or **SIREN** model
- Toggle via the `SIREN` boolean variable

### `src/render_depth_test.py`
- Applied same modifications to `create_nerf()` function  
- Enables model switching during inference

### `src/utils/siren.py` (New)
- Complete **SIREN architecture** implementation
- Follows the structure from `src/utils/nerf_helpers.py`

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data & Checkpoints

```bash
bash download.sh
```

> **Note:** Experiments use the chair scene by default.

---

## Training

### Train with SIREN

**Step 1:** Edit `src/run_platonerf.py`
```python
SIREN = True
```

**Step 2:** Run training
```bash
python src/run_platonerf.py \
    --config configs/chair.txt \
    --downsample 4 \
    --downsample_temp 2 \
    --debug 0 \
    --i_weights 5000 \
    --i_embed -1
```

### Train with NeRF

**Step 1:** Edit `src/run_platonerf.py`
```python
SIREN = False
```

**Step 2:** Run training
```bash
python src/run_platonerf.py \
    --config configs/chair.txt \
    --downsample 4 \
    --downsample_temp 2 \
    --debug 0 \
    --i_weights 5000
```

---

## Inference & Rendering

### Inference with SIREN

**Step 1:** Edit `src/render_depth_test.py`
```python
SIREN = True
```

**Step 2:** Run inference
```bash
python src/render_test_depth.py \
    --config configs/chair.txt \
    --output_dir logs/chair \
    --i_embed -1
```

### 📸 Inference with NeRF

**Step 1:** Edit `src/render_depth_test.py`
```python
SIREN = False
```

**Step 2:** Run inference
```bash
python src/render_test_depth.py \
    --config configs/chair.txt \
    --output_dir logs/chair
```

---

## Checkpoints

All pre-trained checkpoints are located in:
```
logs/chair/
├── nerf_ckpt/      (NeRF checkpoints)
└── siren_ckpt/     (SIREN checkpoints)
```

**To use checkpoints:**
1. Ensure they are properly placed in `logs/chair/`
2. The code will auto-detect them during inference
3. Set the appropriate `SIREN` flag before running
