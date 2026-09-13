# Source Multiplicity Sensitivity in Multisensor Late Fusion

Xianghui Meng &nbsp;·&nbsp; Jionghao Lin

A late-fusion operator receives a list of source posteriors and returns one predictive distribution. Permutation invariance constrains only the order of a *fixed* multiset. It does not specify the fused-output response when the **delivery multiplicity** of an already-existing source changes.

The measured object is the map

$$
k \mapsto F\!\bigl(D_{j,k}(P)\bigr)
$$

where $P=(p_1,\ldots,p_m)$ is held fixed, $p_j$ is delivered $k$ times, every other source is retained once, and no new sensor observation is introduced.

This repository is the reproduction package for that measurement. It is not a new fusion architecture.

> **Signatures.** Product fusion replaces a factor $p_j$ by $p_j^k$. Mean fusion is the count-weighted attraction $F_k-p_j=\frac{m}{k+m-1}(F_1-p_j)$. Under finite-score conditions, normalized token-attention sends duplicate-group mass $A_j(k)\to 1$ and $F\to p_j$.

---

## Layout

```
src/
  models.py, metrics.py, datasets.py
  phase4/                 encoder training + token-attention pooler
  multiplicity_replay/    MHEALTH posterior caches from frozen checkpoints
  multiplicity_sweep/     MHEALTH multiplicity sweep
  hapt_multiplicity/      HAPT loaders, training, and sweep
analysis/                 permutation, collapse, and near-duplicate probes
research/                 frozen checkpoints, caches, and reported tables
data/raw/                 place the public HAR archives here (not shipped)
```

Abandoned earlier pilots (quality-degradation scans, correlated-copy drafts) are not included. The modules above are the ones that generate the reported tables.

Three hash-locked files sit next to the tables because the CLIs refuse to run, or refuse to rewrite a freeze list, if they move: `research/phase4_9of10/05_joint_experiment_outcome.md`, `research/hapt_multiplicity/PROTOCOL.md`, and `research/hapt_multiplicity/sweep/REPLICATION_PROTOCOL.md`. They are frozen experiment records, not extra documentation.

---

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

CPU is sufficient. Apple MPS is used automatically when present. Seeds are $\{0,1,2\}$. The $k$-grid is $1,2,3,5,8,16,32,64,100$.

---

## Data

Download the public archives and place them at

```
data/raw/mhealth.zip
data/raw/hapt.zip
```

- **MHEALTH** (Banos et al.): three wearable placements; subjects $1$–$7$ train, $8$–$10$ test; $2\,\mathrm{s}$ windows, $1\,\mathrm{s}$ hop.
- **HAPT** (Reyes-Ortiz et al.): official $21/9$ subject split; $348$ accelerometer-family and $211$ gyroscope-family features; columns $556$ and $557$ excluded.

The loaders extract the zips on first use. Do not commit the archives.

---

## Reproduction

Frozen checkpoints and posterior caches are already in `research/`. With those files present, the sweeps **do not retrain**.

```bash
# MHEALTH: replay caches, then the exact-copy sweep
python -m multiplicity_replay
python -m multiplicity_sweep

# HAPT: train/freeze only if you are rebuilding encoders; otherwise skip to the sweep
python -m hapt_multiplicity
python -m hapt_multiplicity.sweep

# Controls and the mean-preserving near-duplicate probe
python analysis/permutation_control.py
python analysis/provenance_collapse_control.py
python analysis/near_duplicate_robustness.py
```

Hash gates refuse to overwrite the frozen MHEALTH tree. That is intentional.

### What each script checks

| Script | Object |
| --- | --- |
| `multiplicity_sweep` | $L_1(F_k,F_1)$, flips, mean-fusion identity, token-attention $A_j$ on MHEALTH |
| `hapt_multiplicity.sweep` | the same quantities on HAPT ($m=2$) |
| `permutation_control.py` | reorder a fixed multiset; max elementwise deviation $5.11\times10^{-5}$, no argmax change |
| `provenance_collapse_control.py` | average tokens by source identity; $405$ conditions match clean $k=1$ exactly |
| `near_duplicate_robustness.py` | paired mean-preserving perturbations at target $\eta\in\{0.01,0.05\}$, simplex-capped |

Relative trajectory error is the ratio of *condition-mean* $L_1(F_{\mathrm{near}},F_{\mathrm{exact}})$ to *condition-mean* $L_1(F_{\mathrm{exact}},F_{\mathrm{clean}})$. The movement ratio uses $L_1(\,\cdot\,,F_{\mathrm{clean}})$ in both the numerator and the denominator.

---

## Reported numbers

These values are already in `research/` and should be recovered, not re-tuned.

**Exact copies.** Every clean source and operator produces nonzero $L_1(F_k,F_1)$ on the frozen test posteriors. Mean fusion matches $\|F_k-p_j\|_1=\frac{m}{k+m-1}\|F_1-p_j\|_1$ to a maximum per-example residual of $1.50\times10^{-6}$.

**Near-duplicates** at target $\eta=0.05$:

| | product | mean | attention |
| --- | ---: | ---: | ---: |
| median near/exact movement ratio | $1.0005$ | $1.0000$ | $0.9991$ |
| max condition-level relative trajectory error | $0.2716$ | — | $0.0189$ |

Mean fusion differs from exact duplication by at most $1.86\times10^{-6}$ per-example $L_1$. Product is **not** uniformly equivalent to exact copies; $0.2716$ is the worst condition at $\eta=0.05$.

**Downstream metrics** may improve or degrade. Multiplicity sensitivity is a property of the fused posterior, not a synonym for task harm.

---

## Citation

```bibtex
@article{meng2026multiplicity,
  title   = {Source Multiplicity Sensitivity in Multisensor Late Fusion},
  author  = {Meng, Xianghui and Lin, Jionghao},
  year    = {2026}
}
```
