"""Character-level tokenizer over the Maltese closed character set (T6).

TrOCR ships with a 50 265-token byte-level BPE vocabulary.  The competition
alphabet is a *closed* set of 117 characters (``configs/charset.txt``), so the
Stage-6 decoder swap replaces that vocabulary with this tiny char-level one.

Three special tokens are reserved at the front, followed by the charset in file
order::

    0  <pad>
    1  <bos>
    2  <eos>
    3.. the 117 charset characters

so the full vocabulary is ``117 + 3 = 120`` tokens.  ``save`` / ``load`` round-
trip the exact id assignment: a trained checkpoint must always decode with the
vocabulary it was trained on, or the ids no longer mean anything.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

# Repo root: src/maltese_ocr/train/char_tokenizer.py -> parents[3] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHARSET = _REPO_ROOT / "configs" / "charset.txt"

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


def read_charset(path: str | Path = DEFAULT_CHARSET) -> list[str]:
    """Read ``charset.txt`` into an ordered, de-duplicated list of characters.

    One character per line; a line holding a single space encodes the space
    character, so only the trailing newline is stripped (never all whitespace)
    and genuinely blank lines are dropped.  First-seen order is preserved so the
    id assignment is deterministic across machines.
    """
    chars: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ch = line.rstrip("\n")
            if ch != "":
                chars.append(ch)
    return list(dict.fromkeys(chars))


class CharTokenizer:
    """Map characters to contiguous ids, with <pad>/<bos>/<eos> at ids 0/1/2."""

    def __init__(
        self,
        chars: Iterable[str],
        *,
        pad_token: str = PAD_TOKEN,
        bos_token: str = BOS_TOKEN,
        eos_token: str = EOS_TOKEN,
    ) -> None:
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.chars = list(dict.fromkeys(chars))
        self.id_to_token: list[str] = [pad_token, bos_token, eos_token, *self.chars]
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(self.id_to_token)}
        if len(self.token_to_id) != len(self.id_to_token):
            raise ValueError("duplicate token in vocabulary (a char collides with a special token)")

    @classmethod
    def from_charset_file(cls, path: str | Path = DEFAULT_CHARSET, **kwargs) -> CharTokenizer:
        """Build a tokenizer from a ``charset.txt``-style file."""
        return cls(read_charset(path), **kwargs)

    # -- vocabulary ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def special_ids(self) -> set[int]:
        return {self.pad_token_id, self.bos_token_id, self.eos_token_id}

    # -- encode / decode ----------------------------------------------------

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode ``text`` to ids.  Raises on any out-of-charset character.

        With ``add_special_tokens`` the result is wrapped as ``<bos> … <eos>``.
        Out-of-vocabulary characters fail loudly rather than silently dropping,
        because a silently mangled label would corrupt training invisibly.
        """
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for ch in text:
            try:
                ids.append(self.token_to_id[ch])
            except KeyError as exc:
                raise ValueError(
                    f"character {ch!r} (U+{ord(ch):04X}) is not in the charset"
                ) from exc
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Decode ids back to a string, optionally dropping special tokens."""
        special = self.special_ids
        out: list[str] = []
        for raw in ids:
            i = int(raw)
            if skip_special_tokens and i in special:
                continue
            out.append(self.id_to_token[i])
        return "".join(out)

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the special tokens and char order to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "chars": self.chars,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        """Reconstruct a tokenizer saved with :meth:`save`."""
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            payload["chars"],
            pad_token=payload["pad_token"],
            bos_token=payload["bos_token"],
            eos_token=payload["eos_token"],
        )
