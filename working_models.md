# Trained models — inventory and notes

Renamed copies of every checkpoint live in `models/renamed/`, using a self-describing
naming scheme. The originals are untouched in `models/`; every analysis script
(`D_EnKF.py`, `D_PF.py`, `verify/*`, …) still points at those, so nothing breaks.

## Naming convention

```
{Type}_L{sys}_p{prev}_r{rollout}[_{nonlin}]_t{epoch}
```

| Field | Meaning |
|---|---|
| `Type` | `DenseNN` \| `ResDenseNN` \| `LSTMNN` \| `RNN` |
| `L{sys}` | Lorenz system — `L63`, `L96`, `L05` |
| `p{NN}` | `prev_time_steps` — history window length, zero-padded to 2 |
| `r{NNN}` | `max_rollout_steps` — deepest rollout phase reached in training, zero-padded to 3 so `r005 < r010 < r020 < r100` sorts correctly. `rNA` = the sidecar predates the knob |
| `_tanh` / `_relu` | RNN nonlinearity — **RNN only**; the field exists but is meaningless for the other architectures |
| `t{epoch}` | Unix timestamp — disambiguates runs with identical configs and links back to the original filename and its `runs/` TensorBoard directory |

Example: `ResDenseNN_L63_p04_r020_t1783433501` — residual MLP on L63, 4-step history
window, trained up to 20-step rollouts.

Regenerate the copies and both index files with:

```bash
python tools/rename_models.py            # dry run — prints the mapping, writes nothing
python tools/rename_models.py --apply    # copy into models/renamed/, write MODELS.md
```

`models/renamed/rename_map.json` records the full new → original mapping.

## Inventory (29 models, all L63)

Machine-generated equivalent: `models/MODELS.md`.

### DenseNN — hidden `[64, 64, 32]`, γ=0.95, stateless

| Name | prev | max rollout | Original |
|---|---|---|---|
| `DenseNN_L63_p01_rNA_t1775267887` | 1 | not recorded | `DenseNN_L63_trial1_1775267887` |
| `DenseNN_L63_p02_r020_t1786981313` | 2 | 20 | `DenseNN_L63_trial1_1786981313` |
| `DenseNN_L63_p03_r020_t1786981352` | 3 | 20 | `DenseNN_L63_trial1_1786981352` |
| `DenseNN_L63_p04_r010_t1783435555` | 4 | 10 | `DenseNN_L63_trial1_1783435555` |
| `DenseNN_L63_p04_r020_t1783433501` | 4 | 20 | `DenseNN_L63_trial1_1783433501` |
| `DenseNN_L63_p05_r020_t1786981388` | 5 | 20 | `DenseNN_L63_trial1_1786981388` |
| `DenseNN_L63_p06_r020_t1786986535` | 6 | 20 | `DenseNN_L63_trial1_1786986535` |
| `DenseNN_L63_p10_r010_t1783434502` | 10 | 10 | `DenseNN_L63_trial1_1783434502` |
| `DenseNN_L63_p10_r020_t1783434310` | 10 | 20 | `DenseNN_L63_trial1_1783434310` |

### ResDenseNN — hidden `[64, 64, 32]`, γ=0.95, stateless

| Name | prev | max rollout | Original |
|---|---|---|---|
| `ResDenseNN_L63_p01_rNA_t1775267929` | 1 | not recorded | `ResDenseNN_L63_trial1_1775267929` |
| `ResDenseNN_L63_p02_r020_t1786981468` | 2 | 20 | `ResDenseNN_L63_trial1_1786981468` |
| `ResDenseNN_L63_p03_r020_t1786981516` | 3 | 20 | `ResDenseNN_L63_trial1_1786981516` |
| `ResDenseNN_L63_p04_r010_t1783435555` | 4 | 10 | `ResDenseNN_L63_trial1_1783435555` |
| `ResDenseNN_L63_p04_r020_t1783433501` | 4 | 20 | `ResDenseNN_L63_trial1_1783433501` |
| `ResDenseNN_L63_p05_r020_t1786981559` | 5 | 20 | `ResDenseNN_L63_trial1_1786981559` |
| `ResDenseNN_L63_p10_r010_t1783434506` | 10 | 10 | `ResDenseNN_L63_trial1_1783434506` |
| `ResDenseNN_L63_p10_r020_t1783434297` | 10 | 20 | `ResDenseNN_L63_trial1_1783434297` |

Note: the p=5 sweep has no ResDenseNN counterpart at p=6 — DenseNN has one
(`t1786986535`), ResDenseNN does not.

### LSTMNN — hidden `[64]`, γ=0.95, stateful rollout

| Name | prev | max rollout | Original |
|---|---|---|---|
| `LSTMNN_L63_p01_r005_t1778503362` | 1 | 5 | `LSTMNN_L63_trial1_1778503362` |
| `LSTMNN_L63_p01_r005_t1778600889` | 1 | 5 | `LSTMNN_L63_trial1_1778600889` |
| `LSTMNN_L63_p01_r005_t1778607646` | 1 | 5 | `LSTMNN_L63_trial1_1778607646` |
| `LSTMNN_L63_p01_r005_t1778607685` | 1 | 5 | `LSTMNN_L63_trial1_1778607685` |
| `LSTMNN_L63_p01_r020_t1778510836` | 1 | 20 | `LSTMNN_L63_trial1_1778510836` |
| `LSTMNN_L63_p01_r100_t1779133789` † | 1 | 100 | `LSTMNN_L63_trial1_1779133789` |

The four `p01_r005` checkpoints have byte-identical configs — they differ only in random
seed / run, which is exactly why the timestamp stays in the name.

### RNN — hidden `[64]`, γ=0.95, stateful rollout

| Name | prev | max rollout | Nonlin | Original |
|---|---|---|---|---|
| `RNN_L63_p01_r005_tanh_t1778499778` | 1 | 5 | tanh | `RNN_L63_trial1_1778499778` |
| `RNN_L63_p01_r005_tanh_t1778595234` | 1 | 5 | tanh | `RNN_L63_trial1_1778595234` |
| `RNN_L63_p01_r005_relu_t1778607803` | 1 | 5 | relu | `RNN_L63_trial1_1778607803` |
| `RNN_L63_p01_r005_relu_t1779117546` | 1 | 5 | relu | `RNN_L63_trial1_1779117546` |
| `RNN_L63_p01_r100_tanh_t1779133920` † | 1 | 100 | tanh | `RNN_L63_trial1_1779133920` |
| `RNN_L63_p01_r100_tanh_t1779205626` † | 1 | 100 | tanh | `RNN_L63_trial1_1779205626` |

† Only `_best_model.pth` exists for these — the final-epoch `.pth` was never written
(training was interrupted, or the run was killed after the best checkpoint was saved).
They load fine; `SurrogateModel` picks up the `.yml` sidecar either way.

## Notes

- **Best LSTM so far**: `LSTMNN_L63_p01_r005_t1778503362`
  (was `LSTMNN_L63_trial1_1778503362`). Still plenty to improve.
- **DA baselines**: the two `rNA` models (`DenseNN_L63_p01_rNA_t1775267887`,
  `ResDenseNN_L63_p01_rNA_t1775267929`) are the ones wired into `D_EnKF.py`, `D_PF.py`
  and `D_PF_cupy.py` as the dense/residual baselines. Their sidecars predate the
  `max_rollout_steps` / `rollout_gamma` / `stateful_rollout` knobs, so the depth they
  were trained to is not recorded — at the time it was hardcoded to 5 with γ=0.9, but
  that is inference, not a recorded fact, hence `rNA`.
- **The prev-step sweep** is the `p02`…`p10` DenseNN / ResDenseNN family (timestamps
  `1783…` and `1786…`), all at γ=0.95 and hidden `[64, 64, 32]`, split between
  `r010` and `r020` rollout depths. The matching configs are the `dense*_config.yml` /
  `resdense*_config.yml` files in the repo root.

### RNN-tanh training config (current)

```yml
training:
  num_epochs: 5000
  batch_size: 2048
  learning_rate: 0.001
  n_trials: 1                # independent training runs (each saved separately)
  early_stopping_patience: 20
  early_stopping_min_delta: 1.0e-5  # minimum improvement to reset patience counter
  loss_func: 'MSE'           # 'MSE' | 'Huber'
  split_train: 70            # % of dataset used for training
  split_val: 20              # % for validation  (remainder → test, not used in loop)
  split_test: 10
  max_rollout_steps: 5       # → the r005 tag in the model name
  rollout_gamma: 0.95
  stateful_rollout: true     # thread hidden state through rollout (RNN/LSTMNN only)
  lr_phase_decay: 0.75       # multiply LR by this factor at each phase advance
  lr_scheduler_patience: 10  # ReduceLROnPlateau patience within a phase
  lr_scheduler_factor: 0.5   # ReduceLROnPlateau reduction factor
```

## Known gap

`Main_ML.py` and `1_SingleMLTraining.py` still emit the old
`{Type}_L{sys}_trial{n}_{epoch}` names at training time. Newly trained models therefore
land in `models/` under the old scheme; rerun `python tools/rename_models.py --apply`
to fold them into `models/renamed/` and refresh both index files.
