# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

The project is scaffolded and runnable: `config.py`, `dataset.py`, `model.py`, `train.py`,
`evaluate.py` implement the encoder-only classifier end to end (model forward pass and the
dataset/tokenizer pipeline have both been smoke-tested against the real dataset). Not yet done: a
full training run, hyperparameter tuning, and the class-imbalance-mitigation experiments (weighted
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

Train on the NVIDIA RTX 5070 Ti (CUDA) machine, matching the reference project's
`torch==2.4.0+cu121` pin. The local AMD RX 6650 XT has poor ROCm/DirectML support on Windows for
PyTorch training and should not be used for training runs.

## Commands

- Install dependencies: `pip install -r requirements.txt` (includes a CUDA 12.1 build of
  `torch==2.4.0+cu121` — reinstall the matching CPU/CUDA wheel if the target machine differs; use
  the CUDA machine for real training, see Hardware above).
- Train: `python train.py` (run from this directory — modules import each other by bare name, no
  package structure). Downloads the dataset from Hugging Face on first run (no local corpus file
  needed, unlike the reference project), trains a fresh WordPiece tokenizer to `tokenizer.json`,
  trains the model, and runs a final test-set evaluation (accuracy/macro-F1/confusion matrix)
  printed at the end.
- View training curves: `tensorboard --logdir runs/arxiv_classifier`.
- There is no standalone eval-only entry point yet — `evaluate.py`'s `run_validation`/`run_test`
  are called from within `train.py`; import and call them directly against a loaded checkpoint if
  you need eval without retraining.
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
  - `train_model` — Adam optimizer, plain `CrossEntropyLoss` (no class weighting yet — that's a
    planned follow-up experiment per the proposal, not the baseline), optional checkpoint preload
    (`config["preload"]`), TensorBoard logging, checkpoint saves at epoch 0, the last epoch, and
    every 10th epoch. Runs `evaluate.run_validation` every epoch and `evaluate.run_test` once at
    the end.
- **`evaluate.py`** — `run_validation` (loss/accuracy/macro-F1, logs to TensorBoard, called every
  epoch during training) and `run_test` (accuracy/macro-F1/confusion matrix, called once at the
  end). Written fresh rather than reused from the reference project, which imports these from a
  `test.py` module that doesn't exist in that repo (see its own `CLAUDE.md` for that gap).

## Version control

Not yet a git repository. Plan is to create an empty GitHub repo via the browser, then `git init`
+ push from here — see conversation history for the exact command sequence if it isn't recorded
elsewhere. The user runs all git commands themselves; don't run `git init`/`commit`/`push` here
without being asked.
