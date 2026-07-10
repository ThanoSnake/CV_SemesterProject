# Part 2 — Loss-function experiments (U-Net, Hepatic Vessel)

Studies whether **alternative loss combinations** improve U-Net segmentation over the standard
**Dice + CE** baseline, on the thin, tree-like structures of the Medical Segmentation Decathlon
**Task08 Hepatic Vessel** dataset (where boundary/topology-aware losses are expected to help).

The network, training budget, seed and data are **identical across all loss combos** → a fair,
controlled comparison. Results are reported **before and after** connected-component
post-processing (3D small-object removal).

**Loss combos compared** (`--loss`):
- **`dice_ce`** — baseline (Soft Dice + Cross-Entropy)
- **`ftversky_ce_boundary`** — Focal-Tversky + CE + λ·Boundary (SDF) loss

*(the code also ships `cldice_dice_ce` = clDice + Dice + CE; the default driver runs the two
above, which were the informative comparison on 2D-sliced 3D vessels.)*

---

## How to run — the whole experiment, one command

`run/run_losses.sh` is self-contained: sparse-clones this subfolder → installs deps → downloads
the dataset → preprocesses (boundary SDF maps) → per fold trains + tests each loss → aggregates
+ compares raw and post-processed. GPU VM recommended (torch assumed pre-installed).

```bash
nohup bash run/run_losses.sh &        # progress: tail -f ~/losses-run/run_*.log
```
Default is full 5-fold CV (`FOLDS="0 1 2 3 4"`). Results land in
`<repo>/part2_losses/results/` (and a copy of the log).

---

## How to run — manual, step by step (one fold)

```bash
export TASK=Task08_HepaticVessel
export DATA_DIR=$PWD/data/$TASK        # raw imagesTr/ + labelsTr/ go here

# 1. preprocess: raw NIfTI -> (image, label, 2 boundary-SDF channels) npy + k-fold splits
python3 run_preprocessing_losses.py

# 2. train each loss combo (fold 0)
python3 train_losses.py --loss dice_ce              --fold 0 --out-dir results
python3 train_losses.py --loss ftversky_ce_boundary --fold 0 --out-dir results

# 3. test each (writes RAW and post-processed scores)
python3 test_losses.py --tag dice_ce              --fold 0 --pp-min-size 50 --out-dir results
python3 test_losses.py --tag ftversky_ce_boundary --fold 0 --pp-min-size 50 --out-dir results

# 4. aggregate over folds, then 5. compare vs the dice_ce baseline
python3 train_eval.py --fold-mean dice_ce
python3 train_eval.py --fold-mean ftversky_ce_boundary
python3 train_eval.py --compare dice_ce_mean_scores.json ftversky_ce_boundary_mean_scores.json
```

---

## What each file does, and where its output goes

All outputs are written under **`results/`** (created automatically). `<L>` = loss tag,
`<F>` = fold.

| File | Role | Produces → location |
|---|---|---|
| `run_preprocessing_losses.py` | raw NIfTI → ≥4-channel npy (image+label+2 boundary SDF) + build folds | `data/<TASK>/preprocessed/*.npy`, `data/<TASK>/splits.pkl` |
| `train_losses.py` | train the U-Net for one `--loss` × fold (early-stop on val foreground-Dice) | `results/<L>_f<F>_best.pth`, `_last.pth`, `<L>_f<F>_train.json` (learning curve) |
| `test_losses.py` | full-slice inference → per-class metrics, raw **and** post-processed | `results/<L>_f<F>_scores.json` (raw) and `results/<L>_pp_f<F>_scores.json` (after 3D CC cleanup) |
| `train_eval.py --fold-mean <L>` | mean ± std of each metric across folds | `results/<L>_mean_scores.json` (and `<L>_pp_mean_scores.json`) |
| `train_eval.py --compare a b …` | per-label Dice/ASSD **Δ** of each combo vs the baseline | printed to stdout (no file) |
| `pipeline_loss.py` | shared plumbing: loaders + `set_seed`/`pick_device`/`build_loaders`/`run_epoch`/`run_val_dice`/`evaluate_test` | (imported, no output) |
| `config.py` | paths + task metadata (env-overridable: `TASK`, `DATA_DIR`) | (imported, no output) |

Supporting packages (imported, no direct output): `networks/UNET.py`, `loss_functions/`
(Dice, Focal-Tversky, Boundary, clDice, CE/top-k), `datasets/` (boundary-SDF preprocessing +
loss-aware batchgenerators loaders), `evaluation/` (Dice/ASSD evaluator), `utilities/`.

## Attribution & License
Derived from the MIC-DKFZ [`basic_unet_example`](https://github.com/MIC-DKFZ/basic_unet_example),
© German Cancer Research Center (DKFZ). Apache License 2.0 (see `LICENSE`).
