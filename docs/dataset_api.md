# Dataset API (HuggingFace-like)

This repo ships PDE datasets on disk under:

`datasets/{equation}/ground_truth/*`

The Python API in `eff_physics_learn_dataset.datasets` provides a small HuggingFace-`datasets`-style interface for:
- **seeded, dynamic training budgets** (not hard-coded)
- a **fixed test split** via `test_indices.pkl`
- **parametric interpolation vs extrapolation** splits for few-shot training

## Install / env

```bash
uv sync
```

## Load a dataset

```python
from eff_physics_learn_dataset.datasets import load_pde_dataset

ds = load_pde_dataset("helmholtz2D")  # also: "burgers", "allen_cahn", ...
print(len(ds), ds.param_names, ds.u.shape)
```

Each sample is a dict (JAX-friendly arrays):
- `u`: solution field (e.g. `(64, 64)`)
- `params`: parameter vector `(P,)`
- `param_dict`: named parameters
- plus any available grids (`X_grid`, `Y_grid`, `T_grid`)

```python
sample = ds[0]
print(sample["u"].shape, sample["param_dict"].keys())
```

## Standard modality: seeded budgets + fixed test split

The test split is **authoritative** and comes from:
`datasets/{equation}/ground_truth/test_indices.pkl`.

Budgets are seeded subsamples from the remaining train pool.

```python
splits = ds.budget_split("low", seed=0)  # "low"=25, "medium"=50, "high"=75
train, test = splits["train"], splits["test"]
print(len(train), len(test))
```

Arbitrary budgets (e.g., 10-shot, 37-shot):

```python
splits = ds.train_test_splits(seed=0, n_train=10)
train, test = splits["train"], splits["test"]
```

## Parametric modality: interpolation vs extrapolation (few-shot)

Use this when you want to train on very few samples (e.g., 10–25) and evaluate:
- **Interpolation**: parameter vectors inside the convex hull of the few-shot training parameters
- **Extrapolation**: parameter vectors outside that hull

```python
ps = ds.parametric_splits(seed=0, n_train=10)
train_few = ps["train_few"]
interp = ps["interp"]
extrap = ps["extrap"]
print(len(train_few), len(interp), len(extrap))
```

Notes:
- Candidate pool for `interp`/`extrap` is **all non-training samples** (train pool + fixed test pool).
- Convex-hull membership is computed using `scipy.spatial.Delaunay`; if hull construction fails, it falls back to axis-aligned parameter bounds.

## Output locations (recommended)

By default, analysis/plot scripts in this repo write to:

`docs/_assets/results/{equation}/`

Most scripts support `--out-dir` to override.

## Plot smoke test (25 samples)

Generate a standard 25-sample plot from the **fixed test split**:

```bash
uv run python scripts/plot_dataset_samples.py -e helmholtz2D -s test --n 25
```

This writes (by default):
`docs/_assets/results/helmholtz2D/helmholtz2D_test25.png`

You can also call from Python:

```python
ds.plot_samples(
    split="test",
    n=25,
    seed=0,
    save_path="docs/_assets/results/helmholtz2D/helmholtz2D_test25.png",
)
```

## Plot parameter distributions (scatter)

You can visualize how each split covers parameter space.

Using `plot_param_splits.py`:

```bash
# 2D scatter (P==2) or PCA projection (P>2)
uv run python scripts/plot_param_splits.py -e helmholtz2D -m parametric --n-train 10 --seed 0
```

If the dataset has exactly **3 parameters**, you can request a true **3D scatter**:

```bash
uv run python scripts/plot_param_splits.py -e burgers -m parametric --n-train 10 --seed 0 --projection 3d
```

`solution_similarity_report.py` also writes a param scatter plot by default:
- `docs/_assets/results/{equation}/{equation}_{mode}_param_scatter.png`

Use `--param-projection 3d` for true 3D when P==3.

## Minimal integration patterns

### Parametric Neural Operator (NO) style

Typical training batches look like `(params, u)` pairs:

```python
import numpy as np

splits = ds.train_test_splits(seed=0, n_train=25)
train = splits["train"]

u = np.stack([train[i]["u"] for i in range(len(train))], axis=0)        # (B,H,W)
p = np.stack([train[i]["params"] for i in range(len(train))], axis=0)   # (B,P)
```

### Parametric PINN style

Often you still want `(params, u)` plus a grid to compute PDE residuals:

```python
sample = ds[0]
X = sample.get("X_grid")  # (H,W) if present
Y = sample.get("Y_grid")
T = sample.get("T_grid")
```

## Solution similarity report (interp vs extrap diagnostics)

This is a **diagnostic** to check whether your parametric split also corresponds to solution-space similarity:
- `interp` should typically be **closer** to the training solutions than `extrap`

The report does:
- vectorize solutions into feature vectors
- fit **PCA in solution space**
- compute distances (in PCA space) from each split to the training split (nearest-neighbor + centroid)

### CLI (parametric splits)

```bash
uv run python scripts/solution_similarity_report.py -e helmholtz2D -m parametric --n-train 10 --seed 0
```

Outputs (by default):
- `docs/_assets/results/helmholtz2D/helmholtz2D_parametric_solution_similarity.json`
- `docs/_assets/results/helmholtz2D/helmholtz2D_parametric_solution_similarity.png`
- `docs/_assets/results/helmholtz2D/helmholtz2D_parametric_solution_rows.png`
- `docs/_assets/results/helmholtz2D/helmholtz2D_parametric_param_scatter.png`

### CLI (train/test)

```bash
uv run python scripts/solution_similarity_report.py -e helmholtz2D -m train_test --n-train 25 --seed 0
```

### Python

```python
ps = ds.parametric_splits(seed=0, n_train=10)
report = ds.solution_similarity_report(
    splits={"train_few": ps["train_few"], "interp": ps["interp"], "extrap": ps["extrap"]},
    train_key="train_few",
)
ds.plot_solution_similarity(
    report=report,
    save_path="docs/_assets/results/helmholtz2D/helmholtz2D_parametric_solution_similarity.png",
)
```

### Balanced interp/extrap (e.g. 30/30)

If you want equal-sized `interp` and `extrap`, enable balancing. Note: this **caps** at what’s available (so with very small `n_train` you may not reach 30/30).

```python
ps = ds.parametric_splits(seed=0, n_train=25, balance=True, n_each=30)
print(len(ps["interp"]), len(ps["extrap"]))  # 30, 30
```

CLI:

```bash
uv run python scripts/plot_param_splits.py -e helmholtz2D -m parametric --n-train 25 --seed 0 --balance --n-each 30
uv run python scripts/solution_similarity_report.py -e helmholtz2D -m parametric --n-train 25 --seed 0 --balance --n-each 30
```

### Enforce interp closer than extrap (solution-space ranking)

If you want a *strictly “even comparison”* where `interp` is chosen to be **closest** to training (in solution-PCA space) and `extrap` is chosen to be **farthest**, use:

```bash
uv run python scripts/solution_similarity_report.py -e helmholtz2D -m parametric --n-train 10 --seed 0 --balance --n-each 20 --balance-strategy solution_nn
```

### Improve diversity of the selected extrap samples

Sometimes the “farthest-from-train” region in solution space can still be internally homogeneous. If you want the selected `interp/extrap` subsets to be **more diverse**, add `--diversify` (greedy farthest-point sampling within the top candidates):

```bash
uv run python scripts/solution_similarity_report.py -e burgers -m parametric --n-train 10 --seed 0 --balance --n-each 20 --balance-strategy solution_nn --diversify
```


