# AGENTS.md — working in this repo as an AI agent

Orientation for AI coding agents (and humans who want the fast path). The
authoritative project log is [CLAUDE.md](CLAUDE.md); this file is the "how do I
get a green signal quickly and not trip over the gated bits" cheat sheet.

## Fast feedback loop

```bash
make test          # offline unit tests, ~5s, no network/GPU/HF-login (9 tests)
make pretrain-smoke # real Stage-1 SeqCLR loop, 2 steps, ~10s, offline
```

Run these two before assuming a change is good. `make test` covers the SeqCLR
math (head, NT-Xent loss, checkpoint save/resume) with a tiny CPU stand-in
encoder. `make pretrain-smoke` exercises the *wiring* the unit tests skip:
config load → corpus → `render_pair` → DataLoader → real ViT encoder forward →
loss → optimizer step → checkpoint save.

## What each capability needs

| You want to… | Needs network | Needs HF login | Needs GPU |
|---|---|---|---|
| `make test` (fast unit tests) | no | no | no |
| `make pretrain-smoke` | no¹ | no² | no |
| `make pretrain` (full Stage 1) | base model only¹ | no² | recommended (bf16 path) |
| Stream the real corpus | yes | yes (gated `MLRS/korpus_malti`) | — |
| `make eval` / `test_baseline.py` | no | no | no (needs Tesseract + `mlt`) |

¹ Both load `microsoft/trocr-base-printed` (~1.3 GB) once via the HuggingFace
cache. After the first download they run with `HF_HUB_OFFLINE=1`. `pretrain-smoke`
sets that flag for you.

² Without an HF token, `build_corpus()` logs a warning and falls back to the
local `data/texts.json` paragraphs. That's fine for smoke/dev; real pretraining
wants `huggingface-cli login` for the full corpus.

Force offline explicitly anywhere with `HF_HUB_OFFLINE=1` (and `HF_DATASETS_OFFLINE=1`).

## Pipeline stage status (verify before trusting older notes)

- **Stage 1 — SeqCLR pretrain** (`src/maltese_ocr/pretrain/`): **implemented.**
  `make pretrain` / `make pretrain-smoke`.
- **Stage 2/3 — supervised fine-tune + hard negatives** (`src/maltese_ocr/train/`):
  **stub.** `train/` has only `__init__.py`; `make train` will fail with
  `No module named maltese_ocr.train.run`. `configs/stage2.yaml` / `stage3.yaml`
  exist as forward-declared configs.
- **`make package`**: script is implemented but exits early until a Stage 3
  checkpoint exists (so it's blocked on `make train`).

The current competition submission is the **root-level** Tesseract pipeline
(`competition_transcriber.py`, mean CER **0.0196** over the 422 dev images in
`results.json`) — independent of the `src/` rebuild. Don't delete the root
files; they're the only fully-working path until `src/` reaches parity.

## Conventions

- Run `python3 -m ruff format` and `python3 -m ruff check` before committing.
- Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:` …); one
  logical change per commit.
- **Do not** add a `Co-Authored-By: Claude` trailer — the local git config
  appends one automatically; adding it by hand duplicates it.
- `slow`-marked tests need network/GPU and are skipped by `make test`; run them
  with `make test-all`.

## Gotchas

- Fonts are vendored in `fonts/` (64 files) and loaded via `load_fonts()`. The
  old macOS `/System/Library/Fonts/...` paths in older notes are dead on Linux.
- `data/synthetic*/`, `models/`, and `maltese-vault/` are gitignored —
  `pretrain-smoke` checkpoints (~1.4 GB) write under `models/stage1_smoke/` and
  won't show up in `git status`.
- The repo holds two pipelines side by side: the working root-level Tesseract
  submission and the in-progress `src/maltese_ocr/` staged rebuild.
