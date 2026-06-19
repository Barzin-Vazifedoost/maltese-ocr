"""Stream clean Maltese paragraphs from the MLRS/korpus_malti corpus.

Yields paragraph strings suitable for rendering into synthetic OCR training
images.  Each paragraph is:

* built from one or more consecutive sentences of a single source document
  (split with malti's ``KMSentSplitter``), so it reads coherently;
* filtered to the competition character set (``configs/charset.txt``); and
* length-filtered to ``[min_chars, max_chars]`` characters.

The corpus is gated on HuggingFace — run ``huggingface-cli login`` first.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

# Repo root: src/maltese_ocr/data/corpus.py -> parents[3] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHARSET = _REPO_ROOT / "configs" / "charset.txt"

CORPUS_ID = "MLRS/korpus_malti"


def load_charset(path: str | Path = DEFAULT_CHARSET) -> set[str]:
    """Load the allowed character set, one character per line.

    A blank line encodes the space character, so we read raw bytes and strip
    only the trailing newline rather than all whitespace.
    """
    chars: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            ch = line.rstrip("\n")
            if ch:
                chars.add(ch)
    chars.add(" ")  # space is always allowed
    return chars


def stream_paragraphs(
    *,
    min_chars: int = 10,
    max_chars: int = 500,
    charset_path: str | Path = DEFAULT_CHARSET,
    limit: int | None = None,
) -> Iterator[str]:
    """Yield clean Maltese paragraph strings from MLRS/korpus_malti.

    Args:
        min_chars: drop paragraphs shorter than this.
        max_chars: never let a paragraph grow beyond this many characters.
        charset_path: path to the allowed character set.
        limit: stop after yielding this many paragraphs (None = unbounded).

    Sentences containing any out-of-charset character are skipped (and flush the
    current paragraph buffer), so every yielded character is in the competition
    vocabulary.
    """
    # Imported lazily so importing this module never requires network deps.
    from datasets import load_dataset
    from malti.sent_splitter.km_sent_splitter.km_sent_splitter import KMSentSplitter

    charset = load_charset(charset_path)
    splitter = KMSentSplitter()
    dataset = load_dataset(CORPUS_ID, split="train", streaming=True)

    yielded = 0
    for item in dataset:
        raw = (item.get("text") or item.get("sentence") or "").strip()
        if not raw:
            continue

        buffer: list[str] = []
        buffer_len = 0

        def flush() -> Iterator[str]:
            nonlocal buffer, buffer_len
            if buffer_len >= min_chars:
                yield " ".join(buffer)
            buffer = []
            buffer_len = 0

        for sentence in splitter.split(raw):
            sentence = " ".join(sentence.split())  # normalise whitespace
            if not sentence:
                continue

            # An out-of-charset sentence breaks paragraph continuity.
            if any(ch not in charset for ch in sentence):
                for para in flush():
                    yield para
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return
                continue

            # A single sentence longer than max_chars is yielded on its own
            # (truncation would corrupt the ground truth), provided it is in range.
            if len(sentence) > max_chars:
                for para in flush():
                    yield para
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return
                continue

            extra = 1 if buffer else 0  # joining space
            if buffer_len + extra + len(sentence) > max_chars:
                for para in flush():
                    yield para
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return

            buffer.append(sentence)
            buffer_len += extra + len(sentence)

        for para in flush():
            yield para
            yielded += 1
            if limit is not None and yielded >= limit:
                return
