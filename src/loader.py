from __future__ import annotations
import random
import re
from collections import Counter
from pathlib import Path
from src.config import (
    ENGLISH_DIR,
    RUSSIAN_DIR,
    CHUNK_SIZE,
    MIN_CHUNKS,
    MAX_CHUNKS_PER_AUTHOR,
)


def _safe_dirname(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name).strip("_")


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words) - chunk_size + 1, chunk_size)
    ]


def _load_author_dir(author_dir: Path, chunk_size: int) -> list[str]:
    chunks: list[str] = []
    for f in sorted(author_dir.glob("*.txt")):
        chunks.extend(
            _chunk_text(f.read_text(encoding="utf-8", errors="ignore"), chunk_size)
        )
    return chunks


def _resolve_author_dir(author: str) -> Path | None:
    for base in (ENGLISH_DIR, RUSSIAN_DIR):
        d = base / _safe_dirname(author)
        if d.exists() and any(d.glob("*.txt")):
            return d
    return None


def get_all_disk_authors() -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for base, lang in [(ENGLISH_DIR, "EN"), (RUSSIAN_DIR, "RU")]:
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if d.is_dir() and any(d.glob("*.txt")):
                result.append((d.name.replace("_", " ").strip(), lang))
    return result


def load_corpus(
    chunk_size: int = CHUNK_SIZE,
) -> tuple[list[str], list[str]]:
    bases = [ENGLISH_DIR, RUSSIAN_DIR]
    texts, labels = [], []
    for base in bases:
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            chunks = _load_author_dir(d, chunk_size)
            if len(chunks) < MIN_CHUNKS:
                continue
            if MAX_CHUNKS_PER_AUTHOR and len(chunks) > MAX_CHUNKS_PER_AUTHOR:
                chunks = random.Random(42).sample(chunks, MAX_CHUNKS_PER_AUTHOR)
            display = d.name.replace("_", " ").strip()
            texts.extend(chunks)
            labels.extend([display] * len(chunks))
    return texts, labels


def get_author_statistics(author: str) -> dict:
    d = _resolve_author_dir(author)
    if d is None:
        return {}

    all_text = " ".join(
        f.read_text(encoding="utf-8", errors="ignore") for f in sorted(d.glob("*.txt"))
    )
    if not all_text.strip():
        return {}

    words = [w.lower() for w in re.findall(r"\b\w+\b", all_text)]
    sentences = [s.strip() for s in re.split(r"[.!?]+", all_text) if s.strip()]
    if not words:
        return {}

    sent_word_lens = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    word_lens = [len(w) for w in words]
    punct_count = sum(1 for c in all_text if c in ".,!?;:—–-()[]\"'")

    word_freq = Counter(words)
    n_words = len(words)
    f_of_f = Counter(word_freq.values())
    M2 = sum(v * (k**2) for k, v in f_of_f.items())
    yules_k = max(10_000 * (M2 - n_words) / max(n_words**2, 1), 0.0)

    return {
        "total_words": n_words,
        "unique_words": len(set(words)),
        "total_sentences": len(sentences),
        "yules_k": round(yules_k, 2),
        "punctuation_density": round(punct_count / len(all_text), 5),
        "avg_word_length": round(sum(word_lens) / len(word_lens), 3),
        "avg_sentence_length_chars": round(
            sum(len(s) for s in sentences) / max(len(sentences), 1), 2
        ),
        "_sentence_lengths": sent_word_lens,
        "_word_lengths": word_lens,
    }
