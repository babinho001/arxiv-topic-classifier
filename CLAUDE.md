# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

The project is scaffolded and runnable: `config.py`, `dataset.py`, `model.py`, `train.py`,
`evaluate.py` implement the encoder-only classifier end to end, and `predict.py`,
`dataset_report.py`, `export_sample.py` cover inference and report-asset generation (see
"Report & submission assets" below). All of it has been smoke-tested end to end on a tiny data
subset (training loop → checkpoint → report plots/JSON → `predict.py` loading that checkpoint) and
the dataset-only scripts have been run against the real, full dataset. Not yet done: a full training
run at real scale, hyperparameter tuning, and the class-imbalance-mitigation experiments (weighted
loss / balancing) called for in the proposal text.

`PROJECT_HANDOFF.md` is still the authoritative record of *why* things are the way they are
(dataset analysis, label scheme derivation, proposal text) — read it for background, but for
current implementation status trust the code and this file over its "Open items" list, which
predates scaffolding and is now stale.

There is no git repository yet (see "Version control" below), no test suite, and no lint config.

## What this project is

A SIBP (LLM course) homework: encoder-only Transformer, built from scratch (own tokenizer trained
on own data, weights trained from initialized state — no pretrained/finetuned models), for
single-label topic classification of arXiv paper abstracts (`TimSchopf/arxiv_categories` on
Hugging Face) into a chosen subset of the 123 leaf categories. See `PROJECT_HANDOFF.md` for the
full dataset analysis, label-scheme tradeoffs, and the finalized proposal text (Serbian, 3
sections: Opis problema / Skup podataka / Metode za resavanje problema).

## Reference project

A sibling project at `D:\fakultet\master\SIBP\transformer\transformer\` (**not** inside this
folder — a separate sibling directory) is an earlier from-scratch PyTorch encoder-decoder
Transformer (English→ASL-gloss translation), documented in its own `CLAUDE.md`. This project reuses
its architecture *style* — flat module layout, `InputEmbeddings` / `PositionalEncoding` /
`MultiHeadAttentionBlock` / `FeedForwardBlock` / `EncoderBlock` / `Encoder` built from primitive
`nn.Module`s, a `get_config()` dict as single source of hyperparameters, a `WordLevel` tokenizer
trained from scratch per run — but **drops the decoder entirely** and replaces `ProjectionLayer`
with a pooling + `nn.Linear` classification head, since this is classification, not generation.

Known gap in the reference project worth knowing about if borrowing its `train.py` structure: it
imports `run_validation`/`run_test`/etc. from a `test.py` module that doesn't exist in that repo.
This project's evaluation code (accuracy, macro-F1, confusion matrix) is written fresh regardless.

## Hardware

The RTX 5070 Ti machine originally planned for training is no longer available. Local CPU (Ryzen 5
7600X, 6c/12t) was benchmarked directly against the real model/config and measured **~66 min/epoch**
(~4.7s/training batch at `batch_size=64`, `context_size=256`) — too slow for iterating on the
class-imbalance experiments. Current plan: **train on Google Colab's free GPU notebooks** instead
(see "Training on Colab" below) — Kaggle was considered first but rejected, since enabling its free
GPU tier requires phone/identity verification; `num_epochs` was dropped from 20 to 8 partly to keep
a full run comfortably short even in a worst case. The local AMD RX 6650 XT GPU has poor
ROCm/DirectML support on Windows for PyTorch and should not be used for training.

## Training on Colab

No code changes are needed to run on Colab — clone the GitHub repo into the notebook and run
`train.py` as-is; `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` in `train.py`
already picks up the GPU automatically.

1. New notebook at colab.research.google.com → **Runtime → Change runtime type → Hardware
   accelerator → GPU** (free tier gives a T4). Internet access is on by default, unlike Kaggle —
   no separate toggle needed.
2. In a cell:
   ```
   !git clone https://github.com/<user>/<repo>.git
   %cd <repo>
   !pip install -q datasets tokenizers scikit-learn
   !python train.py
   ```
   Skip installing `requirements.txt` as-is — it pins a specific CUDA build of torch (`+cu121`)
   that may not match Colab's preinstalled torch/driver combination; Colab images already ship a
   working CUDA-enabled torch, tensorboard, and tqdm, so only the few packages not preinstalled
   (`datasets`, `tokenizers`, `scikit-learn`) need installing.
3. Colab's VM is **ephemeral** — `weights/` and `runs/` are lost when the runtime disconnects or
   recycles unless you download them first (file browser sidebar, or mount Google Drive and copy
   the folders there before the session ends). Unlike Kaggle's "Output" panel, nothing persists
   automatically.

Not yet benchmarked on an actual Colab GPU — recommend a quick 1-epoch smoke run first to sanity
check timing before committing to a full unattended run, given free-tier session/quota limits.

## Commands

- Install dependencies: `pip install -r requirements.txt` (includes a CUDA 12.1 build of
  `torch==2.4.0+cu121` — reinstall the matching CPU/CUDA wheel if the target machine differs; use
  the CUDA machine for real training, see Hardware above).
- Train: `python train.py` (run from this directory — modules import each other by bare name, no
  package structure). Downloads the dataset from Hugging Face on first run (no local corpus file
  needed, unlike the reference project), trains a fresh WordPiece tokenizer to `tokenizer.json`,
  trains the model, and runs a final test-set evaluation (accuracy/macro-F1/confusion matrix)
  printed at the end.
- View training curves live: `tensorboard --logdir runs/arxiv_classifier` (static PNG versions are
  also saved to `reports/` automatically at the end of training — see below).
- Generate dataset-only report assets (no training/tokenizer needed): `python dataset_report.py`.
- Export the small sample for the submission zip: `python export_sample.py` (writes
  `sample_data/sample_abstracts.csv`, one row per class, 8 rows — see "Report & submission assets").
- Run inference on a trained model, no retraining: `python predict.py -t "abstract text"` (repeat
  `-t` for multiple, or `-f file.csv`/`-f file.txt`). Needs `tokenizer.json` and a checkpoint under
  `weights/` to already exist (i.e. `train.py` must have run at least once, or those two files
  copied in from elsewhere) — see "Report & submission assets" for exactly what to keep.
- No test suite or linter is configured.

## Architecture

Flat module layout, same style as the reference project — no package structure, modules import
each other by bare name (`from dataset import ...`), so scripts must run with this directory as
the working directory.

- **`config.py`** — single source of hyperparameters/paths (`get_config()`), plus
  `get_weights_file_path`/`get_latest_weights` checkpoint helpers (same scheme as the reference
  project: `{model_basename}{epoch:02d}.pt`). Also defines `TOP_CATEGORIES` (the fixed, ordered
  list of the 8 classes — see PROJECT_HANDOFF.md for how these were chosen) and the
  `CATEGORY_TO_ID`/`ID_TO_CATEGORY` mappings derived from it. Locked-in choices baked into the
  defaults: `context_size=256` (covers ~96% of abstracts untruncated), WordPiece tokenizer
  (`tokenizer_vocab_size=16000`), mean-pooling classification head — see conversation/decision
  rationale in PROJECT_HANDOFF.md if it gets recorded there, otherwise these are simply the
  current config values.
- **`dataset.py`**:
  - `primary_category(categories)` — takes a paper's category list, returns the leaf name of the
    *first* entry (the single-label approximation the proposal commits to).
  - `load_data(config)` — loads `TimSchopf/arxiv_categories` from Hugging Face directly (no TMX
    file, unlike the reference project), filters every split down to rows whose primary category
    is in `TOP_CATEGORIES`, and attaches an integer `label` column.
  - `AbstractDataset` (Torch `Dataset`) — tokenizes one abstract per item as
    `[SOS] tokens [EOS] [PAD]...` to a fixed `context_size`. Abstracts longer than
    `context_size - 2` tokens are **truncated**, not rejected (differs from the reference
    project's `BilingualDataset`, which raises `ValueError` on overlong sentences — abstract
    length is long-tailed enough that erroring would drop usable data). Also builds `encoder_mask`
    (padding-only, shape `(1, 1, context_size)`) — there is no decoder/causal mask, since the
    decoder was dropped entirely.
- **`model.py`** — same primitive `nn.Module` building blocks as the reference project
  (`InputEmbeddings`, `PositionalEncoding`, `LayerNormalization`, `MultiHeadAttentionBlock`,
  `FeedForwardBlock`, `ResidualConnection`, `EncoderBlock`/`Encoder`), assembled by
  `build_encoder_classifier(...)`. The reference project's `Decoder`/`DecoderBlock` and
  `ProjectionLayer` (linear→vocab→log-softmax) are replaced by `ClassificationHead`, which
  mean-pools the encoder's per-token output over non-padding positions (masked average, using
  `encoder_mask`) and projects to `num_classes` raw logits (no log-softmax — `train.py` uses
  `nn.CrossEntropyLoss`, which applies log-softmax internally). `EncoderClassifier.forward(x,
  mask)` is a single end-to-end call (embed → positional encoding → encoder → head), unlike the
  reference `Transformer`'s separate `encode`/`decode`/`project` methods, since there's no
  decode/generate step to keep separate.
- **`train.py`** — orchestrates everything:
  - `get_or_build_tokenizer` trains a **WordPiece** tokenizer from scratch on the abstract text
    (special tokens `[UNK] [PAD] [SOS] [EOS]`, case preserved — `BertNormalizer(lowercase=False)`,
    since acronyms like "CNN" carry topical signal — trained only on the *training* split to avoid
    vocabulary leakage), cached to `tokenizer.json` unless `force_rewrite=True` (current default in
    `get_dataset`, so it's always rebuilt on every run, same behavior as the reference project).
  - `get_dataset` calls `load_data`, wraps each of the dataset's existing train/validation/test
    splits in `AbstractDataset` (no `random_split` needed — the splits come pre-defined, unlike the
    reference project's TMX corpus), and returns `DataLoader`s.
  - `train_model` — AdamW optimizer (`learning_rate=1e-4`, `weight_decay=0.01`), plain
    `CrossEntropyLoss` (no class weighting yet — that's a planned follow-up experiment per the
    proposal, not the baseline). **Tuning note:** the first real run (plain Adam, no weight decay,
    `lr=3e-4`) overfit almost immediately — best validation macro-F1 was epoch 0 of 12, train loss
    collapsed toward 0 while validation loss climbed every epoch after. Switched to AdamW +
    weight decay and dropped the learning rate specifically to fight that (not yet re-validated
    with a full run as of this note). Optional checkpoint preload
    (`config["preload"]`), TensorBoard logging. Two independent checkpoint-saving mechanisms: (1)
    milestone saves at epoch 0, the last epoch, and every 10th epoch, numbered
    `{model_basename}{epoch:02d}.pt`; (2) a **best-checkpoint** save, `{model_basename}best.pt`,
    written every time validation macro-F1 improves on the previous best — independent of the
    milestone schedule, so the best-performing epoch is never lost even if `num_epochs` turns out
    to be more than needed and later epochs overfit. `config.get_best_weights`/`get_latest_weights`
    read these back (`get_latest_weights` explicitly ignores the non-numeric `best` file when
    finding "the latest numbered epoch"). Runs `evaluate.run_validation` every epoch, accumulates a
    `metrics_history` list (train loss/val loss/val accuracy/val macro-F1 per epoch) and writes it
    to `reports/metrics_history.json` after *every* epoch (not just at the end — protects progress
    against a Colab disconnect mid-run), then at the very end calls `evaluate.plot_training_curves`
    and `evaluate.run_test(..., output_dir=reports_dir)`.
- **`evaluate.py`** — `run_validation` (loss/accuracy/macro-F1, logs to TensorBoard, called every
  epoch during training) and `run_test` (accuracy/macro-F1/confusion matrix; called once at the
  end of training, but also usable standalone against any DataLoader + loaded checkpoint since it
  only needs `model`/`dataloader`/`device`, not a training loop). Written fresh rather than reused
  from the reference project, which imports these from a `test.py` module that doesn't exist in
  that repo (see its own `CLAUDE.md` for that gap). Also holds the plotting/report helpers used by
  both `train.py` and `dataset_report.py`: `save_json`, `plot_confusion_matrix`,
  `plot_per_class_metrics`, `plot_training_curves` — all headless (`matplotlib.use('Agg')`), all
  save-to-file only, never display interactively. `run_test`'s optional `output_dir` param, when
  given, saves `confusion_matrix.png`, `per_class_metrics.png`, and `test_results.json` (accuracy,
  macro-F1, full confusion matrix, per-class precision/recall/F1/support, and the sklearn
  `classification_report` text) there.
- **`predict.py`** — standalone inference, no retraining: `load_for_inference` loads
  `tokenizer.json` + a checkpoint (prefers `get_best_weights`, falling back to
  `get_latest_weights` under `weights/` if no best checkpoint exists yet; `--checkpoint` picks one
  explicitly) and rebuilds the model from `get_config()` + the tokenizer's vocab size;
  `predict(text, ...)` mirrors `AbstractDataset.__getitem__`'s tokenize/truncate/pad logic exactly
  (so inference matches training) and returns softmax class probabilities sorted descending. CLI
  accepts repeated `-t/--text`, or `-f/--file` (a `.csv` with an `abstract` column, or one abstract
  per line). This is what satisfies the professor's "enable inference testing, on one or more
  examples from the dataset or outside it" requirement — see "Report & submission assets" below.
- **`dataset_report.py`** — dataset-only report plots/stats, runnable *before* any training (needs
  only the HF dataset download, no tokenizer/model): `dataset_class_distribution.png` (grouped bar
  chart, per-class counts across train/val/test), `dataset_abstract_length_hist.png` (histogram of
  whitespace-word abstract lengths on the train split, with a line marking the `context_size`
  truncation budget — explicitly labeled as a whitespace-word proxy, not real tokenizer token
  counts, since no tokenizer exists yet at this point), and `dataset_stats.json` (row/class counts
  per split, length percentiles, truncation percentage).
- **`export_sample.py`** — writes `sample_data/sample_abstracts.csv`: one example row per class (8
  rows) pulled from the *test* split, for the submission zip. Test split is used because it's
  unseen during training either way, so it doubles as a fair `predict.py` demo set.

## Report & submission assets

The professor's submission email specifies: the zip must contain the report + code + a ~10-row
dataset sample (**not** the full dataset), inference must be testable at the defense (which may
happen on a faculty machine with **no GPU**), and the model must be saved "in a suitable format
(e.g. h5) to run it."

How each requirement is covered here:

- **Sample dataset for the zip**: `python export_sample.py` → `sample_data/sample_abstracts.csv`
  (8 rows, one per class, not gitignored — see `.gitignore`). Run once before zipping.
- **Inference testable without retraining**: `predict.py`, covered above — needs only two saved
  files: `tokenizer.json` and a checkpoint under `weights/`. Both are plain outputs of a `train.py`
  run; nothing else is required. Test it with `python predict.py -f sample_data/sample_abstracts.csv`
  (dataset examples) or `-t "..."` with arbitrary outside text.
- **CPU-only defense machines**: `predict.py` (like `train.py`) auto-selects
  `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`, so it works unmodified on a
  no-GPU faculty machine — inference on a single abstract is fast even on CPU (unlike training).
- **"Suitable format (e.g. h5)"**: h5/HDF5 is a Keras/TensorFlow convention, not applicable here —
  this is PyTorch. `train.py` already saves checkpoints via `torch.save({'model_state_dict': ...})`
  to `weights/*.pt`, which is the PyTorch-native equivalent and is what `predict.py` loads. No
  format conversion needed; if asked at the defense, this is the answer.
- **Report figures/tables**: `dataset_report.py` (dataset-level: class distribution, abstract
  length) and a full `train.py` run (training curves, confusion matrix, per-class
  precision/recall/F1, all under `reports/`, gitignored — regenerate by rerunning rather than
  committing). `reports/test_results.json` also has the full `classification_report` text, handy
  for a results table in the report document itself.

## Version control

This is a GitHub-backed git repository (pushed by the user, who runs all git commands themselves —
don't run `git init`/`commit`/`push` here without being asked). `CLAUDE.md` and `PROJECT_HANDOFF.md`
are intentionally tracked (they're documentation, not secrets). `.gitignore` excludes regenerable
training/report artifacts (`weights/`, `runs/`, `reports/`, `tokenizer.json`, `*.pt`) but *not*
`sample_data/`, which is meant to ship in the submission zip.
