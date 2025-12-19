# Split modalities (paper-style description)

We consider a dataset \(\mathcal{D}=\{(\mathbf{p}_i, \mathbf{u}_i)\}_{i=1}^N\), where \(\mathbf{p}_i\in\mathbb{R}^P\) denotes a vector of PDE parameters (e.g., coefficients, boundary-condition scalars) and \(\mathbf{u}_i\) denotes the associated ground-truth solution field (e.g., on a fixed grid). We expose two complementary evaluation modalities. **Standard modality** uses an authoritative, fixed test split \(\mathcal{T}\subset\{1,\dots,N\}\) (provided as `test_indices.pkl`) to define a stable benchmark across methods; the remaining indices form the train pool \(\mathcal{R}=\{1,\dots,N\}\setminus\mathcal{T}\). For a given random seed \(s\) and budget \(n\), we sample a *seeded* training set \(\mathcal{S}_{n,s}\subset\mathcal{R}\) with \(|\mathcal{S}_{n,s}|=n\), and evaluate on \(\mathcal{T}\), yielding reproducible “few/medium/high data” comparisons while keeping the test distribution unchanged.

In contrast, **parametric modality** targets generalization in parameter space under strict data scarcity. Given a seed \(s\) and a small training size \(n\), we first select a few-shot training set \(\mathcal{F}_{n,s}\subset\mathcal{R}\) with \(|\mathcal{F}_{n,s}|=n\). All remaining samples (including the fixed test set) form a candidate pool \(\mathcal{C}=\{1,\dots,N\}\setminus\mathcal{F}_{n,s}\). We then partition \(\mathcal{C}\) into **interpolation** and **extrapolation** subsets based on convex-hull membership of parameters:
\[
\mathcal{C}_{\mathrm{interp}} = \{ j\in\mathcal{C} : \mathbf{p}_j \in \mathrm{conv}(\{\mathbf{p}_i : i\in\mathcal{F}_{n,s}\}) \},
\qquad
\mathcal{C}_{\mathrm{extrap}} = \mathcal{C}\setminus\mathcal{C}_{\mathrm{interp}}.
\]
Intuitively, \(\mathcal{C}_{\mathrm{interp}}\) evaluates performance on parameter settings that lie inside the region spanned by the few-shot training parameters (in-distribution *within* the sampled parameter simplex), while \(\mathcal{C}_{\mathrm{extrap}}\) evaluates performance on parameter settings outside that region (out-of-distribution in parameter space). This modality separates “generalization by interpolation” from “generalization by extrapolation” without changing the underlying solution representation \(\mathbf{u}\), and is especially relevant for amortized or conditional models that consume \(\mathbf{p}\) as an input.

In practice, one may also want **balanced** interpolation/extrapolation evaluations, i.e. \(|\mathcal{C}_{\mathrm{interp}}|=|\mathcal{C}_{\mathrm{extrap}}|=m\), to control for unequal set sizes. The library supports (i) random balancing and (ii) a solution-aware balancing strategy (`solution_nn`) that ranks candidates by distance to the few-shot training solutions in a low-dimensional solution embedding (PCA of vectorized \(\mathbf{u}\)); it then selects an interpolation subset that is *closest* to the training set and an extrapolation subset that is *farthest* (a stricter separation in solution space). Optionally, `--diversify` applies a greedy farthest-point selection within the top-ranked candidates to encourage *internal diversity* of the chosen subsets (reducing redundancy when “farthest” samples cluster together).

Example (parametric modality; balanced \(m=20\); solution-aware ranking; diversified selection):

```bash
uv run python scripts/solution_similarity_report.py -e flow_mixing -m parametric --n-train 10 --seed 0 --balance --n-each 20 --balance-strategy solution_nn --diversify
```
