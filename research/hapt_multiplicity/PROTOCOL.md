# GATE 5 preregistered HAPT protocol

Written before any HAPT encoder/CPSA training and before inspection of newly trained HAPT test results.

Protocol ID: `hapt_candidate_a_clean_two_source_v1`

## Dataset

Local archive `data/raw/hapt.zip` SHA-256 `4ac4ae064227c07045a99551876b54d204837e995e298b6488559e209b3deb09`.

Local documentation title: Smartphone-Based Recognition of Human Activities and Postural Transitions Data Set, Version 2.1.

Authoritative feature list: `data/raw/hapt_extracted/features.txt`, SHA-256 `bd297980ae769f576fe3b796936b17542e8825df967d6f475ac6214b31bc2fc5`.

Feature map written at preregistration: `research/hapt_multiplicity/feature_map.json`, SHA-256 `f1deccb0aa31346c40a9483f15c502b07dd1957099a909a0357e864e2818a4c2`.

## Sources

This experiment is two-source only. It is not a three-modality dataset. `other` is not a modality. The 348-dimensional group is **accelerometer-family engineered features**, not raw accelerometer waveforms.

- M1: 348 accelerometer-family engineered columns
- M2: 211 gyroscope-family engineered columns
- Excluded exactly: `tBodyGyro-AngleWRTGravity-1` (index 556), `tBodyGyroJerk-AngleWRTGravity-1` (index 557)

Identity: 348 + 211 + 2 = 561. Index sets are disjoint. Union plus the two exclusions equals all official columns.

Grouping does **not** use `src.datasets._group_hapt_features`.

## Split

Official released Train/Test files only. Expected 7767 / 3162 windows, 21 / 9 subjects, empty subject intersection, 12 classes, official labels 1–12 mapped by minus one to 0–11.

## Preprocessing

`dataset_provided_feature_scaling = true`

`additional_fitted_scaling = false`

No z-standardization, min-max fitting, PCA, label-based feature selection, test-statistic normalization, or class resampling. Released features are bounded in [-1, 1]. Whether the original authors fitted that bound only on training subjects is **UNVERIFIED** from local documentation and is not claimed.

## Seeds and holdout

Seeds `{0, 1, 2}`.

For each seed, permute official **training windows** with `numpy.random.default_rng(seed)`. Hold out `int(0.2 * n_train)` windows for CPSA. The complementary 80% trains both encoders. This is **not** independent-subject validation because official HAPT windows overlap by 50%. The official test set never enters encoder training, CPSA holdout, or any fitted preprocessing.

## Models

Unimodal encoders: same MLP as MHEALTH, `d → 128 → 128 → 12` ReLU. Adam, lr `1e-3`, 25 epochs, batch 128, CrossEntropyLoss. Seed convention: M1 uses `seed`, M2 uses `seed + 1`. Device: CPU. No HAPT hyperparameter search. No test-based early stopping.

CPSA: same `ContextualPSA` architecture (expected 1009 parameters). Clean training tokens are exactly `[p1, p2]` on the holdout. No modality IDs, no duplicate tokens, no corrupted tokens. Adam, lr `1e-3`, 25 epochs, batch 128, seed `seed + 10`, CPU. The frozen Phase-4 three-token trainer is not modified.

## Clean evaluation only

Evaluate M1, M2, late-product, late-mean, and two-token CPSA on the official test set. No `k > 1`.

Metrics: accuracy, ECE-15, NLL, multiclass Brier, mean confidence, confidence − accuracy, mean entropy, plus descriptive macro-F1, balanced accuracy, and per-class recall.

## Freeze and replay

Freeze protocol, feature map, three checkpoints per seed, clean metrics, and environment. Replay is inference-only from official HAPT files plus frozen checkpoints and the frozen feature map. Gate-2 match categories apply. PASS requires zero FAIL cells.
