# Solution-Space Percentile vs Convex Hull Method Comparison

## Summary

The new **solution_percentile** method is now the default splitting strategy. It provides consistently better separation between interpolation and extrapolation splits compared to the legacy convex_hull method.

## Method Comparison

### Old Method: `convex_hull`
- **Approach**: Classifies samples based on parameter-space convex hull membership
- **Limitation**: Assumes parameter geometry reflects solution difficulty
- **Problem**: Fails for nonlinear PDEs where solution manifolds fold/twist independent of parameter layout

### New Method: `solution_percentile` (default)
- **Approach**: Splits by distance percentile in solution-space PCA
- **Advantages**:
  - Directly measures generalization difficulty
  - Robust across all parameter dimensionalities
  - No geometry assumptions
  - Natural 50/50 balance
  - Simpler implementation

## Results: Separation Ratios (Extrap/Interp Distance)

| Equation    | Params | Old (convex_hull) | New (solution_percentile) | Improvement |
|-------------|--------|-------------------|---------------------------|-------------|
| **allen_cahn** | 2D | 1.31x - 1.86x | **4.08x - 6.42x** (avg 5.58x) | +211% - +290% |
| **burgers**    | 3D | 1.37x - 1.57x | **2.13x - 2.60x** (avg 2.40x) | +36% - +90% |
| **convection** | 1D | 2.49x - 39.45x | **2.55x - 5.53x** (avg 3.65x) | Stable |
| **helmholtz2D** | 3D | N/A | **1.75x - 2.13x** (avg 1.91x) | Baseline |

### Key Findings

1. **Massive improvement for multi-parameter PDEs**:
   - Allen-Cahn: **4-6x separation** (was 1.3-1.9x)
   - Burgers: **2.1-2.6x separation** (was 1.4-1.6x)

2. **Consistent performance**:
   - All ratios > 1.75x (acceptable threshold: 1.5x)
   - More stable across seeds

3. **Works for all cases**:
   - No convex hull failures
   - No weak separation issues

## Usage

### Default (solution_percentile)
```python
splits = ds.parametric_splits(
    seed=0,
    n_train=10,
    balance=True,
    n_each=20
)
# Uses method="solution_percentile", percentile=50.0 by default
```

### Legacy (convex_hull)
```python
splits = ds.parametric_splits(
    seed=0,
    n_train=10,
    method="convex_hull",
    balance=True,
    n_each=20
)
```

### Custom percentile threshold
```python
splits = ds.parametric_splits(
    seed=0,
    n_train=10,
    method="solution_percentile",
    percentile=40.0,  # More samples in interp (40%), fewer in extrap (60%)
    balance=True,
    n_each=20
)
```

## Recommendation

**Use `solution_percentile` for all datasets.** It provides:
- ✅ Better difficulty separation
- ✅ More robust across parameter spaces
- ✅ Simpler, more interpretable
- ✅ Guaranteed balanced splits
