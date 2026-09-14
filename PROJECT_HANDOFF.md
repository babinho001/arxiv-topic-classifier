# SIBP Homework — Project Handoff / Context Restore

> **For a new Claude Code session in the new project folder:** read this file first, then continue
> from "Open items" below. It captures every decision made in the prior conversation so context
> isn't lost when switching from this folder to the new project folder.

## Course context

- Course: SIBP (large language models), self-directed/mentored. Materials on MS Teams.
- Homework = pick a dataset + task, preprocess it, build an encoder and/or decoder Transformer,
  train it, and test its quality. All choices must be justified at the oral defense.
- **Cannot modify already-finished/pretrained models** — must build the architecture yourself
  (tokenizer trained on your own data, model trained from initialized weights). Note: in the
  proposal document itself, avoid explicitly saying "trained from scratch" — the reviewers aren't
  interested in that framing, just describe the method itself.
- Proposal submission (before starting real work, timing per email instructions): email to
  `stefan.tubic@etf.rs` and `mdodovic@etf.rs`, subject
  `SIBP - prijava domaceg zadatka - 2526 - <GGGG/BBBB Ime Prezime> - <Naslov domaceg zadatka>`,
  with a ≤1-page PDF (same filename as subject) containing exactly three sections: **"Opis
  problema"**, **"Skup podataka"** (must include the dataset download link), **"Metode za
  resavanje problema"**.
- Still unknown: whether the defense happens on our computers or theirs; defense may include a
  live modification task in addition to Q&A.

## Reference project (this folder, `transformer/`)

An existing from-scratch PyTorch Transformer (encoder-decoder) for English→ASL-gloss translation
lives in `transformer/` here (`config.py`, `dataset.py`, `model.py`, `train.py`), documented in
`transformer/CLAUDE.md`. The new project reuses its architecture style (`InputEmbeddings`,
`PositionalEncoding`, `MultiHeadAttentionBlock`, `FeedForwardBlock`, `EncoderBlock`/`Encoder`) but
**drops the decoder entirely** — see Methods below.

## Chosen topic

**Topic classification of arXiv paper abstracts** using an **encoder-only** Transformer (no
decoder needed — this is classification, not generation) plus a small classification head.

### Dataset

- **[`TimSchopf/arxiv_categories`](https://huggingface.co/datasets/TimSchopf/arxiv_categories)**
  on Hugging Face.
- ~204,118 rows total. Fields: `id`, `title`, `abstract`, `categories`, `creation_date`.
- Already split: train 163,168 / validation 20,396 / test 20,397 — no need to `random_split`
  like the reference project does.
- `categories` is a **list** per paper (multi-label), hierarchical strings like
  `"Physics Archive->astro-ph->astro-ph.GA"` (archive → subfield → leaf code). Depth varies
  (some are just `"Computer Science Archive->cs.CV"`, 2 levels).
- Only **7 top-level archives** exist (Physics, Mathematics, Computer Science, Statistics,
  Electrical Engineering/Systems Science, Quantitative Biology, Economics) — Physics alone is
  ~57% of the dataset, so archive-level classification is too coarse/trivial.
- **123 distinct leaf categories** (e.g. `cs.CV`, `hep-ph`) — this is the level we classify at.

### Label scheme decision

- **Single-label**: take the *first* category in each paper's list as its one true label
  (approximation — some papers genuinely have 2+ correct categories, ignored for simplicity).
- Restrict to the **N most frequent leaf categories**, drop every row whose primary label isn't
  in that set. Real measured numbers:

  | N classes | Rows kept | Smallest class | Largest class | Imbalance ratio |
  |---|---|---|---|---|
  | 8  | 65,841  | 6,195 | 11,655 | 1.9x |
  | 10 | 76,871  | 4,994 | 11,655 | 2.3x |
  | 15 | 100,132 | 4,316 | 11,655 | 2.7x |
  | 20 | 118,815 | 3,542 | 11,655 | 3.3x |
  | 25 | 133,084 | 2,479 | 11,655 | 4.7x |
  | 30 | 144,458 | 2,141 | 11,655 | 5.4x |
  | 40 | 160,602 | 1,190 | 11,655 | 9.8x |
  | 50 | 171,906 | 1,037 | 11,655 | 11.2x |

  **Key finding**: more classes = *worse* imbalance (the big classes stay fixed size, you only
  add smaller tail classes), but *more topical diversity* (fewer classes = physics-dominated: at
  N=8 it's 7 physics + 1 CS class; by N=20 it's 12 physics/4 CS/4 math).
- Top-8 by raw frequency (tentative, not finalized): `hep-ph` (11,655), `astro-ph` (11,005),
  `cs.CV` (8,765), `cond-mat.mes-hall` (7,507), `quant-ph` (7,331), `cond-mat.mtrl-sci` (6,768),
  `hep-th` (6,615), `gr-qc` (6,195) → 65,841 rows total (52,671/6,584/6,586 train/val/test).
- **Open decision**: keep this frequency-based top-8 (physics-heavy, low imbalance), or
  hand-pick a more topically diverse set (e.g. mixing in `cs.LG`, `cs.AI`, `cs.CL`, `math.AP`,
  `math.CO`) at the cost of somewhat worse balance. Latest plan (per proposal text): test
  multiple configurations (class balancing, loss weighting, different class counts/choices) to
  address the dominant-class problem rather than pre-committing to one scheme.

### Hardware

- User's own GPU: AMD RX 6650 XT — poor ROCm/DirectML support on Windows for PyTorch training,
  not recommended for this.
- Backup: friend's NVIDIA RTX 5070 Ti (CUDA) — matches the reference project's
  `torch==2.4.0+cu121` pin, use this for actual training.

### Architecture plan (not yet implemented)

- Reuse from `model.py`: `InputEmbeddings` + `PositionalEncoding` → stack of `EncoderBlock`s →
  `Encoder`. **Delete the decoder side entirely.**
- Replace `ProjectionLayer` with a classification head: pool encoder output
  (mean-pool over non-padding positions, or a `[CLS]` token) → `nn.Linear(model_dimension,
  num_classes)`.
- Loss: `CrossEntropyLoss` (single-label). If later extended to true multi-label, switch to
  `BCEWithLogitsLoss` + sigmoid.
- `context_size`: abstracts are longer than the original ASL-gloss sentences — plan for
  128–256 tokens, not the original project's 64.
- Tokenizer: same `WordLevel` approach as the reference `train.py`, but abstract vocabulary is
  larger/noisier than ASL gloss — likely need higher `min_frequency` / capped `vocab_size`, or
  consider switching to a WordPiece/BPE tokenizer (still trained from scratch on this corpus, not
  a pretrained tokenizer).
- Evaluation: the reference project's `test.py` (BLEU/greedy-decode translation testing) doesn't
  apply — need to write equivalent eval reporting accuracy, macro-F1, and a confusion matrix
  across the chosen classes, to make the class-imbalance discussion concrete at the defense.
- Note: reference project's `train.py` currently has a **known bug/gap** — it imports from a
  `test.py` module that doesn't exist in the repo. Not directly relevant to the new project since
  we're writing new eval code anyway, but worth knowing if reusing any of that file's structure.

## Latest proposal document draft (Serbian, ≤1 page, 3 sections)

Still needs: final `<Naslov domaceg zadatka>` (working title used below), student name +
`GGGG/BBBB` index number, and reconciling the class-count/list decision above before this is
final.

**Naslov (radni):** Klasifikacija naučnih radova sa arXiv-a po oblasti korišćenjem enkoder dela
transformera

### Opis problema

Cilj domaćeg zadatka je izgradnja modela koji, na osnovu apstrakta naučnog rada, automatski
prepoznaje kojoj tematskoj oblasti rad pripada (npr. računarski vid, teorijska fizika visokih
energija, kvantna optika...). Ovakav sistem je koristan jer platforme poput arXiv-a svakodnevno
primaju veliki broj radova, pa automatsko tematsko razvrstavanje olakšava pretragu, organizaciju i
preporuku radova bez ručnog označavanja.

Za rešavanje problema koristi se isključivo enkoder deo transformer arhitekture (bez dekodera),
pošto zadatak ne zahteva generisanje novog teksta, već samo razumevanje i klasifikaciju
postojećeg — pristup analogan modelima poput BERT-a. Problem je definisan kao klasifikacija sa
jednom oznakom po primeru (single-label): svaki apstrakt se svrstava u tačno jednu, unapred
definisanu kategoriju iz manjeg, odabranog skupa najzastupljenijih kategorija.

### Skup podataka

Koristi se skup podataka **arXiv Categories**, dostupan na Hugging Face-u:
https://huggingface.co/datasets/TimSchopf/arxiv_categories

Skup sadrži oko 204.000 naučnih radova sa arXiv-a, sa poljima naslov (title), apstrakt (abstract)
i kategorije (categories) — pri čemu svaki rad može pripadati jednoj ili više kategorija,
organizovanih hijerarhijski (npr. `Physics Archive -> astro-ph -> astro-ph.GA`). Skup već dolazi
podeljen na trening (163.168), validacioni (20.396) i test (20.397) podskup.

Za potrebe ovog zadatka, iz liste kategorija svakog rada uzima se prva (primarna) kategorija kao
jedina tačna oznaka, a zadatak se dodatno ograničava na 8 najzastupljenijih ovako dobijenih
kategorija (npr. `cs.CV`, `hep-ph`, `astro-ph`, `quant-ph`...), čime se dobija oko 65.800 radova
ukupno, sa relativno uravnoteženom raspodelom po klasama (odnos najveće i najmanje klase je
približno 1,9:1).

### Metode za resavanje problema

Za rešavanje problema koristiće se enkoder deo transformer arhitekture. Tekst apstrakta se
pretvara u niz numeričkih reprezentacija reči, koje model obrađuje kroz mehanizam samo-pažnje
(self-attention), učeći koje reči i njihovi odnosi unutar apstrakta ukazuju na pripadnost
određenoj kategoriji. Na osnovu ovako naučene reprezentacije, model daje predikciju kategorije
kojoj rad pripada.

Model se trenira na trening skupu, uz periodičnu proveru na validacionom skupu, a konačna ocena
kvaliteta vrši se na test skupu koji model nije video tokom treniranja, merenjem tačnosti
klasifikacije.

Kako je jedna od kategorija (fizika) znatno zastupljenija od ostalih, testiraće se različite
konfiguracije modela i podataka (npr. balansiranje klasa, prilagođavanje funkcije greške, izbor i
broj kategorija) kako bi se ublažio uticaj dominantne klase na kvalitet klasifikacije.

## Open items (pick up here in the new session)

1. Finalize class scheme: top-8-by-frequency (physics-heavy) vs. hand-picked diverse set.
2. Fill in title, name, index number for the email subject + document header.
3. Start scaffolding the new project (config/dataset/model/train, adapted per Architecture plan
   above) once the proposal is submitted/approved.
4. Confirm submission timing per the email instructions (proposal can be sent "any time").
