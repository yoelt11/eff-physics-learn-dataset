#!/bin/bash
# Regenerate all parametric splits using solution_percentile method

set -e

for eq in convection burgers allen_cahn; do
  for seed in 0 1 2; do
    echo "=== Processing $eq seed $seed ==="
    uv run python scripts/solution_similarity_report.py \
      -e "$eq" \
      --seed "$seed" \
      --n-train 10 \
      --balance \
      --n-each 20 \
      --method solution_percentile
  done
done

echo ""
echo "=== Regenerating dataset summary ==="
uv run python scripts/generate_dataset_summary.py

echo ""
echo "✓ All splits regenerated with solution_percentile method"
