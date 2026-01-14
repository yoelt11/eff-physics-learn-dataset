# Split modalities (paper-style description)

We consider a dataset $\mathcal{D}=\{(\mathbf{p}_i, \mathbf{u}_i)\}_{i=1}^N$, where $\mathbf{p}_i\in\mathbb{R}^P$ denotes a vector of PDE parameters (e.g., coefficients, boundary-condition scalars[...] 

In contrast, **parametric modality** targets generalization in parameter space under strict data scarcity. Given a seed $s$ and a small training size $n$, we first select a few-shot training set [...] 
$$
\mathcal{C}_{\mathrm{interp}} = \{ j\in\mathcal{C} : \mathbf{p}_j \in \mathrm{conv}(\{\mathbf{p}_i : i\in\mathcal{F}_{n,s}\}) \},
\qquad
\mathcal{C}_{\mathrm{extrap}} = \mathcal{C}\setminus\mathcal{C}_{\mathrm{interp}}.
$$
Intuitively, $\mathcal{C}_{\mathrm{interp}}$ evaluates performance on parameter settings that lie inside the region spanned by the few-shot training parameters (in-distribution *within* the sampled [...] 

In practice, one may also want **balanced** interpolation/extrapolation evaluations, i.e. $|\mathcal{C}_{\mathrm{interp}}|=|\mathcal{C}_{\mathrm{extrap}}|=m$, to control for unequal set sizes. The l[...] 

Example (parametric modality; balanced $m=20$; solution-aware ranking; diversified selection):

```bash
uv run python scripts/solution_similarity_report.py -e flow_mixing -m parametric --n-train 10 --seed 0 --balance --n-each 20 --balance-strategy solution_nn --diversify
```
