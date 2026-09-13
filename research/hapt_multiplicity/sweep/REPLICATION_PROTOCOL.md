# GATE 6 pre-registered replication criteria

Written before any HAPT \(k>1\) multiplicity computation.

Replication is judged at the **operator-mechanism level**, not by universal task-performance deterioration. Accuracy, macro-F1, ECE, NLL, and Brier are not required to worsen.

Inputs are frozen Gate-5 HAPT test posteriors and seed-specific HAPT CPSA checkpoints only. No training. No \(k>1\) result in this file was used to write these claims.

## Sources and operators

- M1 = accelerometer-family engineered-feature posterior
- M2 = gyroscope-family engineered-feature posterior
- \(k\in\{1,2,3,5,8,16,32,64,100\}\), where \(k\) is the **total** number of exact copies of the duplicated source
- \(k=1\): clean two-token multiset \([p_1,p_2]\)
- Operators: existing `late_product`, existing `late_mean`, frozen two-token-trained HAPT CPSA

## R1 — output sensitivity

For both clean HAPT sources and all three fusion operators, changing source multiplicity may change the fused posterior even though the added tokens are exact numerical copies of one source posterior. Evaluation uses per-example \(L_1(F_k,F_1)\) and prediction-flip rate, not only accuracy or calibration.

## R2 — late-mean finite-k law

For the two-source clean baseline \(P=(p_j,p_o)\),

\[
F_k=\frac{k p_j+p_o}{k+1},\qquad
F_k-p_j=\frac{p_o-p_j}{k+1},\qquad
F_1-p_j=\frac{p_o-p_j}{2}.
\]

Hence for the \(L_1\) norm,

\[
\|F_k-p_j\|_1=\frac{2}{k+1}\|F_1-p_j\|_1.
\]

This is an implementation check of the exact operator law. Required numerical tolerance: maximum per-example residual \(\le 10^{-5}\).

## R3 — late-product concentration

The Gate-3 LP-limit proposition is dataset-independent:

\[
F_k(c)\propto \tilde p_j(c)^k\prod_{i\neq j}\tilde p_i(c).
\]

If \(\tilde p_j\) has a unique maximum class, the mathematical limit is one-hot on that class. HAPT is required only to show finite-\(k\) alignment (rising source-top agreement, falling entropy / rising max-prob). \(k=100\) is not the mathematical limit.

## R4 — CPSA duplicate takeover

The Gate-1 CPSA theorem applies for \(m=2\): \(F(D_{j,k}(P))\to p_j\) as \(k\to\infty\). Finite-\(k\) HAPT evaluation reports duplicate-group attention mass \(A_j(k)\) and \(\|F_k-p_j\|_1\). No theorem-prescribed rate. Finite-\(k\) monotonicity is not required; any non-monotonicity is preserved.

## R5 — task harm is empirical, not guaranteed

Accuracy, macro-F1, ECE, NLL, or Brier need not deteriorate for every source/operator. Improvements are retained and reported.

These five claims will not be edited after the sweep.
