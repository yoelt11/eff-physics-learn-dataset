# Quick Start: Parametric Splits API

## Basic Usage

```python
from eff_physics_learn_dataset import load_pde_dataset

# Load any dataset: allen_cahn, burgers, convection, helmholtz2D
ds = load_pde_dataset("convection")

# Get parametric splits (uses solution_percentile method by default)
splits = ds.parametric_splits(
    seed=0,           # Random seed for reproducibility
    n_train=10,       # Number of training samples
    balance=True,     # Balance interp/extrap splits
    n_each=20         # Samples per split (20 interp, 20 extrap)
)

# Access the three splits
train_ds = splits["train_few"]  # 10 training samples
interp_ds = splits["interp"]     # 20 interpolation samples (easier generalization)
extrap_ds = splits["extrap"]     # 20 extrapolation samples (harder generalization)
```

## Working with Samples

```python
# Iterate over training samples
for sample in train_ds:
    u = sample["u"]              # Solution field (e.g., 64x64 numpy array)
    params = sample["params"]    # Parameter vector (e.g., [23.5])
    param_dict = sample["param_dict"]  # Named dict (e.g., {"beta": 23.5})

    # Your training code here
    model.train(u, params)

# Evaluate on interpolation (easier)
for sample in interp_ds:
    u_pred = model.predict(sample["params"])
    loss_interp = compute_loss(u_pred, sample["u"])

# Evaluate on extrapolation (harder)
for sample in extrap_ds:
    u_pred = model.predict(sample["params"])
    loss_extrap = compute_loss(u_pred, sample["u"])
```

## Dataset Access Methods

```python
# Index access
sample = train_ds[0]
print(sample["u"].shape)          # e.g., (64, 64)
print(sample["param_dict"])       # e.g., {"beta": 23.37}

# Length
print(len(train_ds))              # 10
print(len(interp_ds))             # 20
print(len(extrap_ds))             # 20

# Get all indices
indices = train_ds.indices        # numpy array of original dataset indices
```

## Available Datasets

| Dataset | Parameters | Dims | Separation |
|---------|-----------|------|------------|
| `allen_cahn` | eps, lam | 2D | 5.58x ⭐⭐ |
| `convection` | beta | 1D | 3.65x ⭐⭐ |
| `burgers` | A, k, nu | 3D | 2.40x ⭐ |
| `helmholtz2D` | a1, a2, k | 3D | 1.91x ✓ |

## Advanced Options

### Use Legacy Convex Hull Method

```python
splits = ds.parametric_splits(
    seed=0,
    n_train=10,
    method="convex_hull",  # Legacy parameter-space method
    balance=True,
    n_each=20
)
```

### Custom Percentile Threshold

```python
# Get more interp samples (easier) vs extrap samples (harder)
splits = ds.parametric_splits(
    seed=0,
    n_train=10,
    method="solution_percentile",
    percentile=40.0,  # 40% threshold → more interp, fewer extrap
    balance=True,
    n_each=20
)
```

### Standard Train/Test Split

```python
# Traditional train/test split (no interp/extrap distinction)
splits = ds.train_test_splits(
    seed=0,
    n_train=25
)

train = splits["train"]
test = splits["test"]
```

## What Makes a Good Split?

The **separation ratio** (extrap distance / interp distance) indicates split quality:

- **> 3.0x**: ⭐⭐ Excellent - Very clear difficulty separation
- **2.0-3.0x**: ⭐ Good - Clear separation
- **1.5-2.0x**: ✓ Acceptable - Moderate separation
- **< 1.5x**: ⚠ Weak - Poor separation

Our `solution_percentile` method ensures all datasets have **> 1.75x separation**.

## Example: Full Training Loop

```python
from eff_physics_learn_dataset import load_pde_dataset
import torch

# Setup
ds = load_pde_dataset("convection")
splits = ds.parametric_splits(seed=0, n_train=10, balance=True, n_each=20)

# Training
model = YourNeuralOperator()
for epoch in range(100):
    for sample in splits["train_few"]:
        u = torch.from_numpy(sample["u"])
        params = torch.from_numpy(sample["params"])

        loss = model.train_step(u, params)

    # Evaluate on both splits
    interp_loss = evaluate(model, splits["interp"])
    extrap_loss = evaluate(model, splits["extrap"])

    print(f"Epoch {epoch}: Interp={interp_loss:.4f}, Extrap={extrap_loss:.4f}")
```

## See Also

- [Dataset Summary](datasets/dataset_summary.md) - Full statistics and tables
- [Visualization Index](VISUALIZATION_INDEX.md) - Plot gallery
- [Method Comparison](solution_percentile_comparison.md) - Why solution_percentile?
