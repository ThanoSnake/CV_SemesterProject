# Part 1 — Uncertainty quantification & calibration (MC-Dropout U-Net)

Investigates **predictive uncertainty** of a U-Net on medical image segmentation using
**MC-Dropout**, and whether **calibration** improves it. Two Medical Segmentation Decathlon
datasets are run independently:

- **Hippocampus** (Task04) — files end in **`_hipp`**
- **Spleen** (Task09) — files end in **`_spleen`**

For each dataset two models are trained and compared:

- **`mcdropout`** — baseline MC-Dropout U-Net (loss = Dice + CE)
- **`mcdropout_cal`** — train-time calibrated variant (loss = Dice + CE + λ·soft-binned-ECE)

For every model we report **Dice** (deterministic) and the **uncertainty/calibration** quality
(foreground / macro **ECE**) both **before** and **after** post-hoc **temperature scaling**.

---

## How to run — the whole experiment, one command

Each dataset has a self-contained driver in `run/` that does everything: sparse-clones this
subfolder of the monorepo → installs deps → downloads the dataset → preprocesses → per fold
trains + tests + calibrates + dumps uncertainty for both models → prints a Dice/ECE/temperature
summary. GPU VM recommended (torch assumed pre-installed).

```bash
# Hippocampus (small, 64x64):
nohup bash run/run_all_hippocampus.sh &     # progress: tail -f ~/hippo-run/run_*.log
# Spleen (256x256, heavier):
nohup bash run/run_all_spleen.sh &          # progress: tail -f ~/spleen-run/run_*.log
```
Defaults run **fold 0** only; set `FOLDS="0 1 2 3 4"` for full 5-fold CV. Results land in
`<repo>/part1_uncertainty/results/` (and a copy of the log).

---

## How to run — manual, step by step (one fold)

Set the dataset via env vars, then run the `_hipp` or `_spleen` scripts in this order. Example
for **hippocampus** (swap `_hipp`→`_spleen` and `TASK` for spleen):

```bash
export TASK=Task04_Hippocampus
export DATA_DIR=$PWD/data/$TASK        # raw imagesTr/ + labelsTr/ go here

# 1. preprocess: raw NIfTI -> 2-channel (image,label) npy + k-fold splits
python3 run_preprocessing_mc_hipp.py                 # spleen: run_preprocessing_mc_spleen.py --size 256

# 2. train the two nets (fold 0)
python3 train_mc_hipp.py             --fold 0 --tag mcdropout     --dropout-p 0.4 --out-dir results
python3 train_mc_recalibrate_hipp.py --fold 0 --tag mcdropout_cal --dropout-p 0.4 --cal-weight 1.0 --out-dir results

# 3-6. per net (tag in {mcdropout, mcdropout_cal}) run: test -> uncertainty(raw) -> calibrate -> uncertainty(+T)
python3 test_mc_hipp.py        --fold 0 --tag mcdropout --out-dir results
python3 uncertainty_mc_hipp.py --fold 0 --tag mcdropout --out-dir results --mc-samples 30 --temperature 1.0 --save-volumes
python3 calibrate_mc_hipp.py   --fold 0 --tag mcdropout --out-dir results
python3 uncertainty_mc_hipp.py --fold 0 --tag mcdropout --out-dir results --mc-samples 30 --save-volumes
```

---

## What each file does, and where its output goes

All outputs are written under **`results/`** (created automatically). `<tag>` ∈ {`mcdropout`,
`mcdropout_cal`}, `<F>` = fold.

| File (`_hipp` / `_spleen`) | Role | Produces → location |
|---|---|---|
| `run_preprocessing_mc_*.py` | raw NIfTI → 2-channel `(image,label)` npy + build folds | `data/<TASK>/preprocessed/*.npy`, `data/<TASK>/splits.pkl` |
| `train_mc_*.py` | train baseline MC-Dropout net (Dice+CE) | `results/<tag>_f<F>_best.pth`, `results/<tag>_f<F>_last.pth` |
| `train_mc_recalibrate_*.py` | train calibrated net (Dice+CE+λ·SB-ECE) | `results/mcdropout_cal_f<F>_best.pth`, `_last.pth` |
| `test_mc_*.py` | deterministic (dropout OFF) per-volume metrics | `results/<tag>_f<F>_scores.json` (Dice/ASSD per class) |
| `uncertainty_mc_*.py` | MC sampling (T passes) → entropy / mutual-info / variance maps + ECE | `results/uncertainty/<tag>_f<F>_uncertainty.json` (raw, temp=1) and `..._cal_uncertainty.json` (with fitted T); uncertainty volumes/PNGs under `results/uncertainty/` |
| `calibrate_mc_*.py` | fit a post-hoc **temperature** T on the validation fold | temperature reused by the `+T` uncertainty pass |
| `mc_common_hipp.py` / `mc_common_spleen.py` | shared plumbing: data loaders + `set_seed`/`pick_device`/`run_epoch`/`evaluate_test` | (imported, no output) |
| `config.py` | paths + task metadata (env-overridable: `TASK`, `DATA_DIR`) | (imported, no output) |

Supporting packages (imported, no direct output): `networks/UNET_mc.py` (MC-Dropout U-Net),
`loss_functions/` (Dice, CE, top-k, soft-binned-ECE calibration loss), `datasets/`
(preprocessing + batchgenerators loaders), `evaluation/` (Dice/ASSD evaluator),
`utilities/` (MC-dropout + temperature-calibration helpers).

## Final summary
Both `run/run_all_*.sh` end by printing, across the four settings
(baseline raw / baseline +T / calibrated raw / calibrated +T): mean **Dice**, foreground &
macro **ECE**, and the fitted **temperature**. The reference numbers come from `results/`.

## Attribution & License
Derived from the MIC-DKFZ [`basic_unet_example`](https://github.com/MIC-DKFZ/basic_unet_example),
© German Cancer Research Center (DKFZ). Apache License 2.0 (see `LICENSE`).
