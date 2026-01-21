# Plotting Style Guide for eff-physics-learn-dataset

This project follows the **ePIL-RBF plotting style** for scientific visualizations.

## Standard rcParams Configuration

All plotting functions automatically apply these matplotlib rcParams:

```python
RBF_RCPARAMS = {
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',
    'axes.labelsize': 18,
    'axes.labelweight': 'regular',
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 24,
    'figure.titleweight': 'bold',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'svg.fonttype': 'none'
}
```

## Color Palettes

### ColorBrewer 8 (Colorblind-Friendly)
Used for dataset splits and model comparisons:

- **Train/Train_few**: `#1b9e77` (teal)
- **Interp**: `#7570b3` (purple)
- **Extrap**: `#d95f02` (orange)
- **Test**: `#e7298a` (pink)
- **Standard PINN**: `#66a61e` (green)

### Solution Visualization
- **Solutions**: `rainbow` colormap
- **Errors**: `hot` colormap
- **RBF Kernels**: `viridis` colormap

### Training Curves
- **Loss**: `#f62196` (pink)
- **L2 Error**: `#f3907e` (coral)
- **RL2 Error**: `#00a1d5` (blue)

## Key Styling Rules

1. **Grid**: Always enabled with `alpha=0.3` for subtle appearance
2. **DPI**: All saved figures use 300 DPI
3. **Formats**: Save both PNG and SVG for publication quality
4. **Contours**:
   - 50 levels for filled contours
   - 10 levels for contour lines (black, alpha=0.4, linewidth=0.5)
5. **Titles**: Bold with appropriate font sizes
6. **Aspect Ratio**: Use `'auto'` for data-driven aspect ratios

## Usage

The style is automatically applied when using functions from:
- `src/eff_physics_learn_dataset/datasets/plotting.py`

To manually apply the style in custom scripts:

```python
from eff_physics_learn_dataset.datasets.plotting import maybe_apply_style

maybe_apply_style()
```

## File Output

All plotting functions save both formats:
- **PNG**: 300 DPI, for presentations and documents
- **SVG**: Vector format, for publications and further editing

Example:
```python
fig.savefig('output.png', dpi=300, format='png')
fig.savefig('output.svg', format='svg', dpi=300)
```
