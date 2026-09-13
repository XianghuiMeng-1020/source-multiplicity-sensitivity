# Phase-4 Official Joint Experiment Outcome (Step 5)

**Binding scientific protocol:** `research/phase4_9of10/03b_joint_experiment_protocol_revision.md`  
**Binding implementation audit:** `research/phase4_9of10/04_implementation_semantic_audit.md`  
**Internal name:** `phase4_joint_k_x_degrade_v3_approved`  
**Date:** 2026-09-11  
**This file:** DATA + PRE-REGISTERED VERDICT only. No manuscript rewrite.

`RESULT_FREEZE.sha256` is written after this report is finalized. This file is not edited after that manifest exists.

---

## 1. Run authorization and exact command

External run authorization was granted for Step 5. One official execution was performed.

```
python3 -m src.phase4.run_joint \
  --execute-official \
  --protocol phase4_joint_k_x_degrade_v3_approved
```

Working directory: repository root  
`/Users/mrealsalvatore/Desktop/项目备份/ICASSP 5-9/Idea_8_Evidence_Multiplicity_Bias`

The command was invoked exactly once. It was not invoked a second time.

---

## 2. Pre-run semantic-test result

Command:

```
python3 -m pytest src/phase4/test_joint_semantics.py -v --tb=short
```

Result: **23 passed in 3.06s**. No failure. Official execution was therefore allowed.

Exact pytest output:

```
/opt/homebrew/lib/python3.13/site-packages/pytest_asyncio/plugin.py:208: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture caching scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. Set the default fixture loop scope explicitly in order to avoid unexpected behavior in the future. Valid fixture loop scopes are: "function", "class", "module", "package", "session"

  warnings.warn(PytestDeprecationWarning(_DEFAULT_FIXTURE_LOOP_SCOPE_UNSET))
============================= test session starts ==============================
platform darwin -- Python 3.13.4, pytest-8.3.3, pluggy-1.6.0 -- /opt/homebrew/opt/python@3.13/bin/python3.13
cachedir: .pytest_cache
rootdir: /Users/mrealsalvatore/Desktop/项目备份/ICASSP 5-9/Idea_8_Evidence_Multiplicity_Bias
plugins: anyio-4.12.1, asyncio-0.24.0, cov-6.0.0, typeguard-4.6.0
asyncio: mode=Mode.STRICT, default_loop_scope=None
collecting ... collected 23 items

src/phase4/test_joint_semantics.py::test_h1_parameter_count_exactly_1009 PASSED [  4%]
src/phase4/test_joint_semantics.py::test_h2_k1_permutation_invariance_all_six PASSED [  8%]
src/phase4/test_joint_semantics.py::test_h2_k5_permutation_invariance_sample PASSED [ 13%]
src/phase4/test_joint_semantics.py::test_h3_contextual_capacity_dummy_tokens PASSED [ 17%]
src/phase4/test_joint_semantics.py::test_h4_score_input_is_48d_not_token_independent PASSED [ 21%]
src/phase4/test_joint_semantics.py::test_h5_train_holdout_separation_mock_ids PASSED [ 26%]
src/phase4/test_joint_semantics.py::test_h5_k9_standardize_ignores_test_statistics PASSED [ 30%]
src/phase4/test_joint_semantics.py::test_h6_train_derived_stuck_ignores_any_test_matrix PASSED [ 34%]
src/phase4/test_joint_semantics.py::test_h7_late_product_list_vs_weight_identity PASSED [ 39%]
src/phase4/test_joint_semantics.py::test_h8_arm_cardinalities PASSED     [ 43%]
src/phase4/test_joint_semantics.py::test_h9_same_checkpoint_no_optimizer_on_eval PASSED [ 47%]
src/phase4/test_joint_semantics.py::test_h10_two_token_legality_not_quality PASSED [ 52%]
src/phase4/test_joint_semantics.py::test_h11_no_flags_input_dim_12 PASSED [ 56%]
src/phase4/test_joint_semantics.py::test_h12_write_guard_allows_and_rejects PASSED [ 60%]
src/phase4/test_joint_semantics.py::test_h13_default_cli_locked PASSED   [ 65%]
src/phase4/test_joint_semantics.py::test_h13_subprocess_default_does_not_train PASSED [ 69%]
src/phase4/test_joint_semantics.py::test_h14_cpsa_determinism_cpu_two_epochs PASSED [ 73%]
src/phase4/test_joint_semantics.py::test_h15_historical_csv_unchanged PASSED [ 78%]
src/phase4/test_joint_semantics.py::test_k13_output_schema_dummy_row PASSED [ 82%]
src/phase4/test_joint_semantics.py::test_k16_train_cpsa_cannot_receive_encoders PASSED [ 86%]
src/phase4/test_joint_semantics.py::test_k18_stuck_does_not_take_labels PASSED [ 91%]
src/phase4/test_joint_semantics.py::test_no_official_result_files_exist PASSED [ 95%]
src/phase4/test_joint_semantics.py::test_k17_stuck_uses_train_row_zero_not_a_test_row PASSED [100%]

============================== 23 passed in 3.06s ==============================
```

---

## 3. Pre-run hashes

Recorded before the official command. Official result files were absent (`joint_raw.csv` absent, `joint_summary.csv` absent, `results/` directory absent).

| SHA256 | File |
|---|---|
| `d64bd0e504276baaa53aac759e7a638c9f99c95e40a66dd2dbfd1b23a2c33306` | `research/phase4_9of10/03b_joint_experiment_protocol_revision.md` |
| `436b46518dab0a86c7163bef46dc67c64fa1c76975cf182c0459984a82fc9070` | `research/phase4_9of10/04_implementation_semantic_audit.md` |
| `9dbedbefd34161567431c20727625416937b75dd474537e8125f1c529318380c` | `src/phase4/__init__.py` |
| `d4a1a8f50477ccda582935db1a507be43a87553197f5390ee1d4e52adef22318` | `src/phase4/config.py` |
| `5714421b041c58c9eb9dce69fc94329c66fcf1d41e0c46e8a207fb5a15310056` | `src/phase4/contextual_psa.py` |
| `71e287ae99e5bcfd6c177922d9b126efbbc006a0ccd66859ac6129c44444563c` | `src/phase4/run_joint.py` |
| `1ea41f841c18702f20f8394baf69df333533dc97389b19ff29f57945a6b52be6` | `src/phase4/test_joint_semantics.py` |
| `f0f1023a5c7283219ac229be1d8e7673a3a87700a4197c2c73ba9c41cd7dfa76` | `paper/main.tex` |
| `5050b7f032ca90e9737f4c601c94662bc694b4450a859bf67b3976eeb17eb4d2` | `paper/refs.bib` |

Pre-run historical CSV hashes:

| SHA256 | File |
|---|---|
| `15ee31f666034c65315463f730bec0d218dee2412c80594f7ad2164d429e9f68` | `research/results/metrics.csv` |
| `446771587cccd2d9e3faa02a1e9f26fb08710498f6da47cb64141f79b43c1f6c` | `research/phase2/results/phase2_metrics.csv` |
| `af5759e454b6274cfccec802526d2720329ca28947faf2cbce998719c3501c09` | `research/silent_failure_pilot/results/metrics.csv` |
| `e6cf27c9846178ee72e6c1aa8b04f11bdf63a60dc51f8725a032d642cb5bf5fd` | `research/silent_failure_phase2/results/phase2_metrics.csv` |
| `c5df188d3c06e967eb18c0027a5f5d973e19e076a0f93f2b3e0bbce9c3c3d34b` | `research/silent_failure_phase3_5/results/concat_multiseed_raw.csv` |
| `01165ad4c8ad58e913167383c93e577b0b25bcb6dd83f2de342311304bdb3432` | `research/silent_failure_phase3_5/results/decomposition_multiseed_raw.csv` |
| `84d88ef40a3f7f8f23479777435ea1bf3dafb5a05da462468da332cd78c79671` | `research/silent_failure_phase3_5/results/decomposition_multiseed_summary.csv` |

---

## 4. Exact official stdout/stderr

The official process printed no training logs. Complete captured stdout:

```
===OFFICIAL_EXIT=0===
===OFFICIAL_RUNTIME_SEC=17===
===OFFICIAL_START=1789126473===
===OFFICIAL_END=1789126490===
```

Complete captured stderr: empty.

The four wrapper lines were added by the Step-5 shell around the official module. `src.phase4.run_joint` itself emitted no text on the successful path.

---

## 5. Exit status and runtime

| Item | Value |
|---|---|
| Exit status | **0** |
| Wall-clock runtime | **17 seconds** |
| Start (local) | 2026-09-11T19:34:33 |
| End (local) | 2026-09-11T19:34:50 |
| Unix start/end | 1789126473 / 1789126490 |

The run completed. Evidence validation proceeded.

---

## 6. Files created

Created by the official runner:

| Path | Role |
|---|---|
| `research/phase4_9of10/results/joint_raw.csv` | 30 raw metric rows |
| `research/phase4_9of10/results/config.json` | protocol / dataset / seeds / k_dup |
| `research/phase4_9of10/results/checkpoints/seed{0,1,2}_enc{1,2,3}.pt` | 9 encoder checkpoints |
| `research/phase4_9of10/results/checkpoints/seed{0,1,2}_cpsa.pt` | 3 CPSA checkpoints |

Created after official exit, from immutable `joint_raw.csv` only (no training, no evaluation):

| Path | Role |
|---|---|
| `research/phase4_9of10/results/joint_summary.csv` | across-seed mean/std, `ddof=0` |

Not created by the frozen runner (and not added here):

| Path | Status |
|---|---|
| `research/phase4_9of10/results/protocol_snapshot.md` | **ABSENT** |

`config.json` contents:

```json
{
  "protocol": "phase4_joint_k_x_degrade_v3_approved",
  "dataset": "MHEALTH",
  "seeds": [0, 1, 2],
  "k_dup": 5
}
```

---

## 7. Post-run hashes

Official artifacts immediately after the official process exited:

| SHA256 | File | Bytes |
|---|---|---|
| `be407ae37e8835899048276e50a26ee848f8d26d6d09c8644d6d1980afbdc205` | `research/phase4_9of10/results/joint_raw.csv` | 5258 |
| `de0640a4823e386be80fefd0fab02730394ddef5451dfbf5f3c1e077b64aa941` | `research/phase4_9of10/results/config.json` | 132 |
| `c693087d8ea562c98d732886dcce557380c636d7a494894d7b8e65e99427e36f` | `checkpoints/seed0_enc1.pt` | 83157 |
| `3d942d026ab6162800803be74809386cccc520ff8f4d329a9b87b8d584768696` | `checkpoints/seed0_enc2.pt` | 98517 |
| `f568f9f98594e63d2ea0993e8164d76017d2b29880c3b52eaccf7f9aa9c8cb73` | `checkpoints/seed0_enc3.pt` | 98517 |
| `48cb179a6fc810864358df255ec8b8ff57374906929b2b8a15a8a73ef5706aaf` | `checkpoints/seed0_cpsa.pt` | 6805 |
| `b19570ac18e295338cd3c54a219576945ae5b0509982f72124cb89dc4e311265` | `checkpoints/seed1_enc1.pt` | 83157 |
| `655b35682676c8424b5565a07b29ef54724821e0ac64fa26c9a29de232b48691` | `checkpoints/seed1_enc2.pt` | 98517 |
| `37f3534957b9ccee02f7995cb9f4f6779df189b45995f7b85f3413aea5332420` | `checkpoints/seed1_enc3.pt` | 98517 |
| `19e0b18c069fc47d25e0b16c5f3b9167e2a3e7280976bb26c3b98bea61740b85` | `checkpoints/seed1_cpsa.pt` | 6805 |
| `b043d0a02f8f3f38ddb27a44045f12d14ea6fcb6414870fc054e432401514a03` | `checkpoints/seed2_enc1.pt` | 83157 |
| `6ae1c67d5f21d36d4a46ea5125155b6be1856d532dfe1c290b79a2d7ffc9a457` | `checkpoints/seed2_enc2.pt` | 98517 |
| `8a38f933a20cf93187f239ec592de9395e3db46a17c1a1eeb132e6092c107b7c` | `checkpoints/seed2_enc3.pt` | 98517 |
| `37523b25e98d283d371de8bf26b01333bdf8ac3ff39aea0403badb4d286cf629` | `checkpoints/seed2_cpsa.pt` | 6805 |

Post-processing artifact:

| SHA256 | File | Bytes |
|---|---|---|
| `a310c142925fa123e3e07819e0b70e6fe5e764fa7c3d442ef6545cbc3a67febd` | `research/phase4_9of10/results/joint_summary.csv` | 3361 |

Post-run source / protocol / manuscript hashes (identical to Section 3):

| SHA256 | File |
|---|---|
| `d64bd0e504276baaa53aac759e7a638c9f99c95e40a66dd2dbfd1b23a2c33306` | `research/phase4_9of10/03b_joint_experiment_protocol_revision.md` |
| `436b46518dab0a86c7163bef46dc67c64fa1c76975cf182c0459984a82fc9070` | `research/phase4_9of10/04_implementation_semantic_audit.md` |
| `9dbedbefd34161567431c20727625416937b75dd474537e8125f1c529318380c` | `src/phase4/__init__.py` |
| `d4a1a8f50477ccda582935db1a507be43a87553197f5390ee1d4e52adef22318` | `src/phase4/config.py` |
| `5714421b041c58c9eb9dce69fc94329c66fcf1d41e0c46e8a207fb5a15310056` | `src/phase4/contextual_psa.py` |
| `71e287ae99e5bcfd6c177922d9b126efbbc006a0ccd66859ac6129c44444563c` | `src/phase4/run_joint.py` |
| `1ea41f841c18702f20f8394baf69df333533dc97389b19ff29f57945a6b52be6` | `src/phase4/test_joint_semantics.py` |
| `f0f1023a5c7283219ac229be1d8e7673a3a87700a4197c2c73ba9c41cd7dfa76` | `paper/main.tex` |
| `5050b7f032ca90e9737f4c601c94662bc694b4450a859bf67b3976eeb17eb4d2` | `paper/refs.bib` |

---

## 8. Historical / source integrity check

**SOURCE / PROTOCOL FILES CHANGED DURING RUN = 0**

Post-run historical CSV hashes are identical to the pre-run hashes in Section 3.

**CHANGED HISTORICAL CSV FILES = 0**  
`paper/main.tex` and `paper/refs.bib` unchanged.

`joint_raw.csv` and all official checkpoints are treated as immutable from this point.

---

## 9. joint_raw structural validation

| Check | Result |
|---|---|
| Row count | **30** (3 seeds × 2 rules × 5 arms) |
| Seeds | 0, 1, 2 |
| Rules | `late_product`, `cpsa` |
| Arms | A, B, C, D, E |
| Unique `(seed, rule, arm)` | 30 / 30; no duplicates; no missing cells |
| `k` | A=1, B=5, C=1, D=5, E=0 |
| `reliability` | A/B=`clean`; C/D=`hard_stuck_trainref`; E=`dropped` |
| `n_tokens` | A=3, B=7, C=3, D=7, E=2 |
| NaN / inf | **none** |
| accuracy / confidence / ECE in [0, 1] | **yes** |
| NLL / Brier / entropy | all finite and ≥ 0 |
| `n` | constant **1987.0** (MHEALTH test size) |
| Required metric columns | accuracy, confidence, ece, nll, brier, entropy, conf_acc_gap, n |

Normalization: the frozen runner scores `late_product` of posterior lists and CPSA `sum_i a_i p_i`. Both maps emit class-simplex vectors when the encoder posteriors do. This run did not re-evaluate probabilities. Raw metrics are finite and in-range.

**STRUCTURAL VALIDATION: PASS**

---

## 10. Checkpoint inventory

Expected: 3 encoder checkpoints + 1 CPSA checkpoint per seed = **12** model files. Observed: **12**. No extras.

| File | SHA256 | Trainable tensors | Parameter count |
|---|---|---|---|
| `seed0_enc1.pt` | `c693087d8ea562c98d732886dcce557380c636d7a494894d7b8e65e99427e36f` | `net.{0,2,4}.{weight,bias}` | 20108 |
| `seed0_enc2.pt` | `3d942d026ab6162800803be74809386cccc520ff8f4d329a9b87b8d584768696` | `net.{0,2,4}.{weight,bias}` | 23948 |
| `seed0_enc3.pt` | `f568f9f98594e63d2ea0993e8164d76017d2b29880c3b52eaccf7f9aa9c8cb73` | `net.{0,2,4}.{weight,bias}` | 23948 |
| `seed0_cpsa.pt` | `48cb179a6fc810864358df255ec8b8ff57374906929b2b8a15a8a73ef5706aaf` | `phi.0`, `g.0`, `g.2` | **1009** |
| `seed1_enc1.pt` | `b19570ac18e295338cd3c54a219576945ae5b0509982f72124cb89dc4e311265` | `net.{0,2,4}.{weight,bias}` | 20108 |
| `seed1_enc2.pt` | `655b35682676c8424b5565a07b29ef54724821e0ac64fa26c9a29de232b48691` | `net.{0,2,4}.{weight,bias}` | 23948 |
| `seed1_enc3.pt` | `37f3534957b9ccee02f7995cb9f4f6779df189b45995f7b85f3413aea5332420` | `net.{0,2,4}.{weight,bias}` | 23948 |
| `seed1_cpsa.pt` | `19e0b18c069fc47d25e0b16c5f3b9167e2a3e7280976bb26c3b98bea61740b85` | `phi.0`, `g.0`, `g.2` | **1009** |
| `seed2_enc1.pt` | `b043d0a02f8f3f38ddb27a44045f12d14ea6fcb6414870fc054e432401514a03` | `net.{0,2,4}.{weight,bias}` | 20108 |
| `seed2_enc2.pt` | `6ae1c67d5f21d36d4a46ea5125155b6be1856d532dfe1c290b79a2d7ffc9a457` | `net.{0,2,4}.{weight,bias}` | 23948 |
| `seed2_enc3.pt` | `8a38f933a20cf93187f239ec592de9395e3db46a17c1a1eeb132e6092c107b7c` | `net.{0,2,4}.{weight,bias}` | 23948 |
| `seed2_cpsa.pt` | `37523b25e98d283d371de8bf26b01333bdf8ac3ff39aea0403badb4d286cf629` | `phi.0`, `g.0`, `g.2` | **1009** |

`enc1` is smaller than `enc2`/`enc3` (chest_acc 15-d vs ankle/arm 45-d). CPSA keys match the frozen architecture. Checkpoints were opened only to read `state_dict` keys and element counts. No new test-set inference was run from them.

Absent, as required:

- arm-specific CPSA
- k-specific CPSA
- stuck-specific CPSA
- separate E / drop CPSA

**No extra trained fusion checkpoint. No protocol-violation flag.**

---

## 11. Full 30-row raw table

Values are copied from immutable `joint_raw.csv`.

| arm | seed | rule | k | reliability | n_tokens | accuracy | confidence | ece | nll | brier | entropy | conf_acc_gap | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0 | late_product | 1 | clean | 3 | 0.8761952692501258 | 0.978377103805542 | 0.10218185096394197 | 1.1162067651748657 | 0.22265377640724182 | 0.05402576923370361 | 0.10218183455541618 | 1987 |
| A | 0 | cpsa | 1 | clean | 3 | 0.8782083543029693 | 0.7945723533630371 | 0.10837595490996539 | 0.4643822908401489 | 0.21768373250961304 | 0.5036197304725647 | -0.08363600093993218 | 1987 |
| B | 0 | late_product | 5 | clean | 7 | 0.8309008555611475 | 0.9936884045600891 | 0.16450889310347233 | 3.5018670558929443 | 0.3322047293186188 | 0.016654327511787415 | 0.16278754899894166 | 1987 |
| B | 0 | cpsa | 5 | clean | 7 | 0.7790639154504277 | 0.8693971633911133 | 0.16406416167070592 | 0.7518134117126465 | 0.3806294798851013 | 0.35393407940864563 | 0.09033324794068553 | 1987 |
| C | 0 | late_product | 1 | hard_stuck_trainref | 3 | 0.6678409662808253 | 0.9562907814979553 | 0.2884498179618834 | 2.981703281402588 | 0.5975381135940552 | 0.11428727954626083 | 0.28844981521713 | 1987 |
| C | 0 | cpsa | 1 | hard_stuck_trainref | 3 | 0.7443381982888777 | 0.715195894241333 | 0.11050504297543004 | 0.7484011054039001 | 0.380597859621048 | 0.7111602425575256 | -0.029142304047544698 | 1987 |
| D | 0 | late_product | 5 | hard_stuck_trainref | 7 | 0.16205334675390035 | 0.9731419682502747 | 0.8110886367050079 | 21.402620315551758 | 1.635473608970642 | 0.06093333289027214 | 0.8110886214963743 | 1987 |
| D | 0 | cpsa | 5 | hard_stuck_trainref | 7 | 0.09159536990437846 | 0.8219203352928162 | 0.7303249550093701 | 2.0956807136535645 | 1.2855194807052612 | 0.5410795211791992 | 0.7303249653884377 | 1987 |
| E | 0 | late_product | 0 | dropped | 2 | 0.8268746854554605 | 0.9506626725196838 | 0.12378800584253619 | 0.7303304076194763 | 0.2753427028656006 | 0.1204565018415451 | 0.12378798706422334 | 1987 |
| E | 0 | cpsa | 0 | dropped | 2 | 0.8535480624056366 | 0.8061283826828003 | 0.10516267718684452 | 0.47358664870262146 | 0.23658767342567444 | 0.44106578826904297 | -0.04741967972283634 | 1987 |
| A | 1 | late_product | 1 | clean | 3 | 0.8968293910417715 | 0.9692427515983582 | 0.07381922117597087 | 1.0164449214935303 | 0.17675670981407166 | 0.06339968740940094 | 0.07241336055658665 | 1987 |
| A | 1 | cpsa | 1 | clean | 3 | 0.7805737292400604 | 0.7815080285072327 | 0.08463518592239803 | 0.5260748863220215 | 0.2780533730983734 | 0.511926531791687 | 0.0009342992671722561 | 1987 |
| B | 1 | late_product | 5 | clean | 7 | 0.8379466532460996 | 0.9883230328559875 | 0.1575848253815991 | 3.3457207679748535 | 0.31959831714630127 | 0.024375885725021362 | 0.15037637960988792 | 1987 |
| B | 1 | cpsa | 5 | clean | 7 | 0.7559134373427278 | 0.8722741603851318 | 0.12693138805297255 | 0.8662022352218628 | 0.39787912368774414 | 0.34204423427581787 | 0.11636072304240408 | 1987 |
| C | 1 | late_product | 1 | hard_stuck_trainref | 3 | 0.7005535983895319 | 0.9253622889518738 | 0.22536987223754767 | 2.4206154346466064 | 0.5278217792510986 | 0.1670517772436142 | 0.22480869056234187 | 1987 |
| C | 1 | cpsa | 1 | hard_stuck_trainref | 3 | 0.6411675893306492 | 0.7055924534797668 | 0.20896373313458408 | 0.967993974685669 | 0.4951505959033966 | 0.724289059638977 | 0.06442486414911763 | 1987 |
| D | 1 | late_product | 5 | hard_stuck_trainref | 7 | 0.1716155007549069 | 0.9691108465194702 | 0.7974953362554178 | 20.96170425415039 | 1.6106460094451904 | 0.07968005537986755 | 0.7974953457645633 | 1987 |
| D | 1 | cpsa | 5 | hard_stuck_trainref | 7 | 0.09159536990437846 | 0.9352882504463196 | 0.8436928602337657 | 3.362344980239868 | 1.6349200010299683 | 0.28346017003059387 | 0.8436928805419411 | 1987 |
| E | 1 | late_product | 0 | dropped | 2 | 0.8595873175641671 | 0.9716684222221375 | 0.11489616470176127 | 0.6448432207107544 | 0.2513405382633209 | 0.08128955960273743 | 0.11208110465797039 | 1987 |
| E | 1 | cpsa | 0 | dropped | 2 | 0.6869652742828385 | 0.8132224678993225 | 0.1378233212664551 | 0.6646130681037903 | 0.3825811445713043 | 0.4392054080963135 | 0.12625719361648402 | 1987 |
| A | 2 | late_product | 1 | clean | 3 | 0.9068948163059889 | 0.984604001045227 | 0.08103869758662596 | 0.993664026260376 | 0.17868775129318237 | 0.04346023127436638 | 0.07770918473923816 | 1987 |
| A | 2 | cpsa | 1 | clean | 3 | 0.8459989934574735 | 0.8012821674346924 | 0.05610995827815494 | 0.47982874512672424 | 0.23200847208499908 | 0.4900752007961273 | -0.04471682602278115 | 1987 |
| B | 2 | late_product | 5 | clean | 7 | 0.8399597382989431 | 0.9956936836242676 | 0.15573386772229678 | 3.464202404022217 | 0.31414151191711426 | 0.010359210893511772 | 0.15573394532532447 | 1987 |
| B | 2 | cpsa | 5 | clean | 7 | 0.7710115752390538 | 0.8783818483352661 | 0.1518105027150076 | 0.7943692803382874 | 0.3875395357608795 | 0.3332540988922119 | 0.10737027309621228 | 1987 |
| C | 2 | late_product | 1 | hard_stuck_trainref | 3 | 0.7700050327126321 | 0.9514077305793762 | 0.1821448134308073 | 1.9729808568954468 | 0.41745725274086 | 0.11469583958387375 | 0.18140269786674412 | 1987 |
| C | 2 | cpsa | 1 | hard_stuck_trainref | 3 | 0.6874685455460493 | 0.7337839007377625 | 0.15610602432810367 | 0.6968331336975098 | 0.3522087037563324 | 0.6718181371688843 | 0.04631535519171315 | 1987 |
| D | 2 | late_product | 5 | hard_stuck_trainref | 7 | 0.1716155007549069 | 0.977408766746521 | 0.8057932999785605 | 20.353254318237305 | 1.6287509202957153 | 0.06190621107816696 | 0.805793265991614 | 1987 |
| D | 2 | cpsa | 5 | hard_stuck_trainref | 7 | 0.09159536990437846 | 0.8737099170684814 | 0.7821145939898958 | 2.405987501144409 | 1.4338854551315308 | 0.4479283094406128 | 0.782114547164103 | 1987 |
| E | 2 | late_product | 0 | dropped | 2 | 0.8379466532460996 | 0.9684675335884094 | 0.1329344127216589 | 0.6740245819091797 | 0.28300943970680237 | 0.08331044018268585 | 0.1305208803423098 | 1987 |
| E | 2 | cpsa | 0 | dropped | 2 | 0.7372924006039255 | 0.8022958636283875 | 0.09281600168078881 | 0.5622892379760742 | 0.28796619176864624 | 0.4467024505138397 | 0.06500346302446192 | 1987 |

Clean-arm sanity (seed 0 late-product B vs A vs the historical prior):

| Metric | A | B | Direction | Historical prior |
|---|---|---|---|---|
| accuracy | 0.876195 | 0.830901 | decrease | decrease |
| ECE | 0.102182 | 0.164509 | increase | increase |
| NLL | 1.116207 | 3.501867 | increase | increase |
| Brier | 0.222654 | 0.332205 | increase | increase |

Direction matches the protocol's historical MHEALTH late-product seed-0 k=1 → k=5 prior. See Section 22 for the additional numerical-identity observation.

---

## 12. Across-seed summary table

Created as `joint_summary.csv` from immutable `joint_raw.csv`. Standard deviation uses **ddof = 0**.

| rule | arm | acc mean | acc std | conf mean | conf std | ECE mean | ECE std | NLL mean | NLL std | Brier mean | Brier std | entropy mean | entropy std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| late_product | A | 0.893306 | 0.012778 | 0.977408 | 0.006309 | 0.085680 | 0.012035 | 1.042105 | 0.053217 | 0.192699 | 0.021196 | 0.053629 | 0.008145 |
| late_product | B | 0.836269 | 0.003884 | 0.992568 | 0.003112 | 0.159276 | 0.003777 | 3.437263 | 0.066532 | 0.321982 | 0.007564 | 0.017130 | 0.005732 |
| late_product | C | 0.712800 | 0.042598 | 0.944354 | 0.013576 | 0.231988 | 0.043650 | 2.458433 | 0.412677 | 0.514272 | 0.074139 | 0.132012 | 0.024778 |
| late_product | D | 0.168428 | 0.004508 | 0.973221 | 0.003388 | 0.804792 | 0.005594 | 20.905860 | 0.430218 | 1.624957 | 0.010485 | 0.067507 | 0.008617 |
| late_product | E | 0.841470 | 0.013585 | 0.963600 | 0.009241 | 0.123873 | 0.007364 | 0.683066 | 0.035481 | 0.269898 | 0.013490 | 0.095019 | 0.018006 |
| cpsa | A | 0.834927 | 0.040621 | 0.792454 | 0.008211 | 0.083040 | 0.021367 | 0.490095 | 0.026211 | 0.242582 | 0.025755 | 0.501874 | 0.009006 |
| cpsa | B | 0.768663 | 0.009596 | 0.873351 | 0.003746 | 0.147602 | 0.015449 | 0.804128 | 0.047206 | 0.388683 | 0.007088 | 0.343077 | 0.008474 |
| cpsa | C | 0.690991 | 0.042193 | 0.718191 | 0.011702 | 0.158525 | 0.040232 | 0.804409 | 0.117572 | 0.409319 | 0.061789 | 0.702422 | 0.022294 |
| cpsa | D | 0.091595 | 0.000000 | 0.876973 | 0.046340 | 0.785377 | 0.046340 | 2.621338 | 0.539068 | 1.451442 | 0.143181 | 0.424156 | 0.106508 |
| cpsa | E | 0.759269 | 0.069760 | 0.807216 | 0.004527 | 0.111934 | 0.018988 | 0.566830 | 0.078052 | 0.302378 | 0.060467 | 0.442325 | 0.003187 |

Audited: each mean/std recomputes from the three raw seeds and matches `joint_summary.csv`.

---

## 13. Absolute delta tables

Deltas are arm − reference. Mean and std use ddof = 0.

### Late-product

| Contrast | metric | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|---|
| B−A (dependence) | accuracy | -0.04529441 | -0.05888274 | -0.06693508 | -0.05703741 | 0.00893060 |
| B−A | ece | +0.06232704 | +0.08376560 | +0.07469517 | +0.07359594 | 0.00878670 |
| B−A | nll | +2.38566029 | +2.32927585 | +2.47053838 | +2.39515817 | 0.05805993 |
| B−A | brier | +0.10955095 | +0.14284161 | +0.13545376 | +0.12928211 | 0.01427431 |
| C−E (reliability primary) | accuracy | -0.15903372 | -0.15903372 | -0.06794162 | -0.12866969 | 0.04294123 |
| C−E | ece | +0.16466181 | +0.11047371 | +0.04921040 | +0.10811531 | 0.04716233 |
| C−E | nll | +2.25137287 | +1.77577221 | +1.29895627 | +1.77536712 | 0.38882255 |
| C−E | brier | +0.32219541 | +0.27648124 | +0.13444781 | +0.24437482 | 0.07993917 |
| C−A (reliability secondary) | accuracy | -0.20835430 | -0.19627579 | -0.13688978 | -0.18050663 | 0.03123347 |
| C−A | ece | +0.18626797 | +0.15155065 | +0.10110612 | +0.14630824 | 0.03496424 |
| C−A | nll | +1.86549652 | +1.40417051 | +0.97931683 | +1.41632795 | 0.36188346 |
| C−A | brier | +0.37488434 | +0.35106507 | +0.23876950 | +0.32157297 | 0.05935290 |
| D−A | accuracy | -0.71414192 | -0.72521389 | -0.73527932 | -0.72487838 | 0.00863257 |
| D−A | ece | +0.70890679 | +0.72367612 | +0.72475460 | +0.71911250 | 0.00722995 |
| D−A | nll | +20.28641355 | +19.94525933 | +19.35959029 | +19.86375439 | 0.38273806 |
| D−A | brier | +1.41281983 | +1.43388930 | +1.45006317 | +1.43225743 | 0.01524825 |
| D−B | accuracy | -0.66884751 | -0.66633115 | -0.66834424 | -0.66784097 | 0.00108719 |
| D−B | ece | +0.64657974 | +0.63991051 | +0.65005943 | +0.64551656 | 0.00421093 |
| D−B | nll | +17.90075326 | +17.61598349 | +16.88905191 | +17.46859622 | 0.42597117 |
| D−B | brier | +1.30326888 | +1.29104769 | +1.31460941 | +1.30297533 | 0.00962127 |
| D−C | accuracy | -0.50578762 | -0.52893810 | -0.59838953 | -0.54437175 | 0.03934825 |
| D−C | ece | +0.52263882 | +0.57212546 | +0.62364849 | +0.57280426 | 0.04123982 |
| D−C | nll | +18.42091703 | +18.54108882 | +18.38027346 | +18.44742644 | 0.06827619 |
| D−C | brier | +1.03793550 | +1.08282423 | +1.21129367 | +1.11068446 | 0.07346386 |
| D−E | accuracy | -0.66482134 | -0.68797182 | -0.66633115 | -0.67304144 | 0.01057535 |
| D−E | ece | +0.68730063 | +0.68259917 | +0.67285889 | +0.68091956 | 0.00601425 |
| D−E | nll | +20.67228991 | +20.31686103 | +19.67922974 | +20.22279356 | 0.41083544 |
| D−E | brier | +1.36013091 | +1.35930547 | +1.34574148 | +1.35505929 | 0.00659730 |

### CPSA

| Contrast | metric | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|---|
| B−A (dependence) | accuracy | -0.09914444 | -0.02466029 | -0.07498742 | -0.06626405 | 0.03102735 |
| B−A | ece | +0.05568821 | +0.04229620 | +0.09570054 | +0.06456165 | 0.02268714 |
| B−A | nll | +0.28743112 | +0.34012735 | +0.31454054 | +0.31403300 | 0.02151614 |
| B−A | brier | +0.16294575 | +0.11982575 | +0.15553106 | +0.14610085 | 0.01882428 |
| C−A (reliability primary) | accuracy | -0.13387016 | -0.13940614 | -0.15853045 | -0.14393558 | 0.01056470 |
| C−A | ece | +0.00212909 | +0.12432855 | +0.09999607 | +0.07548457 | 0.05281280 |
| C−A | nll | +0.28401881 | +0.44191909 | +0.21700439 | +0.31431410 | 0.09428683 |
| C−A | brier | +0.16291413 | +0.21709722 | +0.12020023 | +0.16673719 | 0.03965029 |
| C−E (reliability secondary) | accuracy | -0.10920986 | -0.04579768 | -0.04982386 | -0.06827713 | 0.02899044 |
| C−E | ece | +0.00534237 | +0.07114041 | +0.06329002 | +0.04659093 | 0.02934269 |
| C−E | nll | +0.27481446 | +0.30338091 | +0.13454390 | +0.23757975 | 0.07378483 |
| C−E | brier | +0.14401019 | +0.11256945 | +0.06424251 | +0.10694072 | 0.03280734 |
| D−A | accuracy | -0.78661298 | -0.68897836 | -0.75440362 | -0.74333166 | 0.04062078 |
| D−A | ece | +0.62194900 | +0.75905767 | +0.72600464 | +0.70233710 | 0.05842266 |
| D−A | nll | +1.63129842 | +2.83627009 | +1.92615876 | +2.13124242 | 0.51285714 |
| D−A | brier | +1.06783575 | +1.35686663 | +1.20187698 | +1.20885979 | 0.11809962 |
| D−B | accuracy | -0.68746855 | -0.66431807 | -0.67941621 | -0.67706761 | 0.00959594 |
| D−B | ece | +0.56626079 | +0.71676147 | +0.63030409 | +0.63777545 | 0.06166836 |
| D−B | nll | +1.34386730 | +2.49614275 | +1.61161822 | +1.81720942 | 0.49236537 |
| D−B | brier | +0.90489000 | +1.23704088 | +1.04634592 | +1.06275893 | 0.13609578 |
| D−C | accuracy | -0.65274283 | -0.54957222 | -0.59587318 | -0.59939607 | 0.04219283 |
| D−C | ece | +0.61981991 | +0.63472913 | +0.62600857 | +0.62685254 | 0.00611585 |
| D−C | nll | +1.34727961 | +2.39435101 | +1.70915437 | +1.81692833 | 0.43420506 |
| D−C | brier | +0.90492162 | +1.13976941 | +1.08167675 | +1.04212259 | 0.09987248 |
| D−E | accuracy | -0.76195269 | -0.59536990 | -0.64569703 | -0.66767321 | 0.06975992 |
| D−E | ece | +0.62516228 | +0.70586954 | +0.68929859 | +0.67344347 | 0.03480378 |
| D−E | nll | +1.62209406 | +2.69773191 | +1.84369826 | +2.05450808 | 0.46373827 |
| D−E | brier | +1.04893181 | +1.25233886 | +1.14591926 | +1.14906331 | 0.08307033 |

No relative-ECE percentages are used.

---

## 14. Late-product Q1 I2 decision

Predicate (B vs A): `B ECE > A ECE` AND (`B NLL > A NLL` OR `B Brier > A Brier`) AND `B accuracy ≤ A accuracy + 0.005`.

| Seed | Δacc | ΔECE | ΔNLL | ΔBrier | Harmful |
|---|---|---|---|---|---|
| 0 | -0.045294 | +0.062327 | +2.385660 | +0.109551 | YES |
| 1 | -0.058883 | +0.083766 | +2.329276 | +0.142842 | YES |
| 2 | -0.066935 | +0.074695 | +2.470538 | +0.135454 | YES |

Mean deltas: Δacc = **-0.057037**, ΔECE = **+0.073596**, ΔNLL = **+2.395158**, ΔBrier = **+0.129282**.  
Mean effect is in the harmful direction. Harmful seeds: **3 / 3**.

**Late-product Q1 I2: PASS**

---

## 15. Late-product Q2 I2 decision

Predicate (C vs E): `C ECE > E ECE` AND (`C NLL > E NLL` OR `C Brier > E Brier`) AND `C accuracy < E accuracy`.

| Seed | Δacc | ΔECE | ΔNLL | ΔBrier | Harmful |
|---|---|---|---|---|---|
| 0 | -0.159034 | +0.164662 | +2.251373 | +0.322195 | YES |
| 1 | -0.159034 | +0.110474 | +1.775772 | +0.276481 | YES |
| 2 | -0.067942 | +0.049210 | +1.298956 | +0.134448 | YES |

Mean deltas: Δacc = **-0.128670**, ΔECE = **+0.108115**, ΔNLL = **+1.775367**, ΔBrier = **+0.244375**.  
Mean effect is in the harmful direction. Harmful seeds: **3 / 3**.

**Late-product Q2 I2: PASS**

---

## 16. CPSA Q1 I2 decision

Predicate (B vs A): same as late-product Q1.

| Seed | Δacc | ΔECE | ΔNLL | ΔBrier | Harmful |
|---|---|---|---|---|---|
| 0 | -0.099144 | +0.055688 | +0.287431 | +0.162946 | YES |
| 1 | -0.024660 | +0.042296 | +0.340127 | +0.119826 | YES |
| 2 | -0.074987 | +0.095701 | +0.314541 | +0.155531 | YES |

Mean deltas: Δacc = **-0.066264**, ΔECE = **+0.064562**, ΔNLL = **+0.314033**, ΔBrier = **+0.146101**.  
Mean effect is in the harmful direction. Harmful seeds: **3 / 3**.

**CPSA Q1 I2: PASS**

---

## 17. CPSA Q2 I2 decision

Predicate (C vs A): `C ECE > A ECE` AND (`C NLL > A NLL` OR `C Brier > A Brier`) AND `C accuracy ≤ A accuracy + 0.005`.  
Reference remains C vs A. It was not changed after seeing numbers.

| Seed | Δacc | ΔECE | ΔNLL | ΔBrier | Harmful |
|---|---|---|---|---|---|
| 0 | -0.133870 | +0.002129 | +0.284019 | +0.162914 | YES |
| 1 | -0.139406 | +0.124329 | +0.441919 | +0.217097 | YES |
| 2 | -0.158530 | +0.099996 | +0.217004 | +0.120200 | YES |

Mean deltas: Δacc = **-0.143936**, ΔECE = **+0.075485**, ΔNLL = **+0.314314**, ΔBrier = **+0.166737**.  
Mean effect is in the harmful direction. Harmful seeds: **3 / 3**.

Seed 0 satisfies the ECE clause by +0.002129. The NLL and Brier clauses are large. The pre-registered predicate is strict `>` and is met.

**CPSA Q2 I2: PASS**

---

## 18. Outcome class

| Rule | Q1 I2 | Q2 I2 | Class |
|---|---|---|---|
| late-product | PASS | PASS | **LP-P3** |
| CPSA | PASS | PASS | **LF-1** |

---

## 19. Pre-registered merged-paper case

**CASE A**

late-product **LP-P3** + CPSA **LF-1**.

Pre-registered implication: strongest Phase-4 support — both a structural and a contextual learned rule exhibit both single-factor diagnostics.

This is a data label for Step 6. It is not a manuscript rewrite.

---

## 20. Interaction table

\[
I(M)=[M(D)-M(C)]-[M(B)-M(A)]
\]

| rule | metric | seed 0 | seed 1 | seed 2 | mean | std (ddof=0) | descriptive reading |
|---|---|---|---|---|---|---|---|
| late_product | accuracy | -0.46049321 | -0.47005536 | -0.53145445 | -0.48733434 | 0.03144092 | super-additive accuracy drop |
| late_product | ece | +0.46031178 | +0.48835986 | +0.54895332 | +0.49920832 | 0.03699187 | super-additive ECE rise |
| late_product | nll | +16.03525674 | +16.21181297 | +15.90973508 | +16.05226827 | 0.12390805 | super-additive NLL rise |
| late_product | brier | +0.92838454 | +0.93998262 | +1.07583991 | +0.98140236 | 0.06694509 | super-additive Brier rise |
| cpsa | accuracy | -0.55359839 | -0.52491193 | -0.52088576 | -0.53313202 | 0.01456495 | super-additive accuracy drop |
| cpsa | ece | +0.56413171 | +0.59243292 | +0.53030803 | +0.56229089 | 0.02539576 | super-additive ECE rise |
| cpsa | nll | +1.05984849 | +2.05422366 | +1.39461383 | +1.50289533 | 0.41310947 | super-additive NLL rise |
| cpsa | brier | +0.74197587 | +1.01994365 | +0.92614569 | +0.89602174 | 0.11546171 | super-additive Brier rise |

On this metric scale, both rules show super-additive joint harm in D. Interaction is descriptive. It is not a success gate and does not falsify P3. No reliability × independence product was fitted. No post-hoc “near zero” threshold was defined.

---

## 21. CPSA diagnostics if pre-recorded

`joint_raw.csv` has no `m1_attention_mass`, no `|h1-c|`, and no other contextual diagnostic column.

**NOT RECORDED IN PRE-REGISTERED RUN.**

Checkpoints were not reopened to manufacture a new test-set diagnostic.

---

## 22. Unexpected / adverse results

1. **Seed-0 late-product A, B, and E are numerically identical** to historical Idea-8 / Idea-9A MHEALTH late-product rows: unique `k=1`, duplicate `k=5`, and dropped. Accuracy, confidence, ECE, NLL, Brier, and entropy match those frozen CSVs at the printed precision. Phase-4 encoders are trained on an 80% holdout, so exact identity with the historical full-train seed-0 late-product is unexpected. Arm C does **not** match historical 9A `retained_stuck` (0.667841 vs 0.627076), which is the expected consequence of the train-derived stuck rule. Seeds 1 and 2 differ from seed 0. This is flagged for external audit. It is not treated as a structural invalidation. The run was not repeated.

2. **CPSA arm D accuracy is identical on all three seeds:** 0.09159536990437846 (182 / 1987). Confidence / ECE / NLL / Brier still vary by seed.

3. **CPSA Q2 seed 0 ECE delta is +0.002129.** The pre-registered ECE clause is still satisfied. NLL and Brier move substantially.

4. **Official runner stdout is empty** on the success path. Runtime is 17 seconds for 9 encoder trainings + 3 CPSA trainings. That is consistent with the small MLPs and is not a missing-output condition: 30 raw rows and 12 checkpoints were written.

5. **`protocol_snapshot.md` was not emitted** by the frozen runner. See Section 23.

No result was hidden. No seed was replaced.

---

## 23. Protocol deviations

**NONE** relative to 03b scientific decisions and the frozen Step-4 implementation.

`joint_summary.csv` was missing after the official process and was created by deterministic post-processing of immutable `joint_raw.csv`, as this step authorized. `run_joint.py` was not edited.

03b K.1 listed `protocol_snapshot.md` as a later runner artifact. The authorized implementation does not write it. It remains absent.

---

## 24. Number of official training runs

**Exactly 1.**

---

## 25. Number of rerun seeds

**Exactly 0.**

Seeds used: 0, 1, 2. No substitution.

---

## 26. RESULT FREEZE

**PASS**

`joint_raw.csv`, `joint_summary.csv`, `config.json`, and the 12 official checkpoints are immutable. This report is hashed next; `RESULT_FREEZE.sha256` is then written and is not used to edit this file.

---

## 27. READY FOR EXTERNAL SCIENTIFIC AUDIT

**YES**

Pre-registered verdict for that audit:

- Late-product: **LP-P3**
- CPSA: **LF-1**
- Merged-paper case: **A**

Manuscript consequences belong to Step 6.

---

*End of Step-5 official outcome. One official MHEALTH run. No manuscript edit. No protocol edit. No source edit.*
