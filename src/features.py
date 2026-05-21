from __future__ import annotations
import math
import re
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import (
    ENGLISH_FUNCTION_WORDS,
    RUSSIAN_FUNCTION_WORDS,
    SPACY_MODELS,
)

_PUNCT = ".,!?;:—–-()[]\"'"
_UD_TAGS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]

_SPACY_NLP = None


def _load_spacy(language: str):
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    try:
        import spacy

        model = SPACY_MODELS["ru" if language == "ru" else "en"]
        _SPACY_NLP = spacy.load(model, disable=["ner", "lemmatizer"])
    except Exception:
        _SPACY_NLP = "fallback"
    return _SPACY_NLP


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def feature_function_words(texts: list[str], func_words: list[str]) -> np.ndarray:
    fw_idx = {w: i for i, w in enumerate(func_words)}
    out = np.zeros((len(texts), len(func_words)), dtype=np.float32)
    for ti, text in enumerate(texts):
        words = re.findall(r"\b\w+\b", text.lower())
        total = max(len(words), 1)
        cnt = Counter(words)
        for w, i in fw_idx.items():
            out[ti, i] = cnt.get(w, 0) / total
    return out


def feature_hapax_ratio(text: str) -> float:
    words = [w.lower() for w in _tokens(text)]
    if not words:
        return 0.0
    freq = Counter(words)
    hapax = sum(1 for c in freq.values() if c == 1)
    return hapax / max(len(set(words)), 1)


def feature_yules_k(text: str) -> float:
    words = [w.lower() for w in _tokens(text)]
    n = len(words)
    if n == 0:
        return 0.0
    freq = Counter(words)
    M2 = sum(v * (k**2) for k, v in Counter(freq.values()).items())
    return math.log1p(max(10_000 * (M2 - n) / max(n**2, 1), 0.0))


def feature_ttr(text: str, sample_size: int | None = 1000) -> float:
    words = [w.lower() for w in _tokens(text)]
    if not words:
        return 0.0
    sample = words[:sample_size] if sample_size else words
    return len(set(sample)) / max(len(sample), 1)


def feature_avg_word_length(text: str) -> float:
    words = _tokens(text)
    return sum(len(w) for w in words) / max(len(words), 1)


def feature_long_word_ratio(text: str) -> float:
    words = _tokens(text)
    return sum(1 for w in words if len(w) > 6) / max(len(words), 1)


def feature_short_word_ratio(text: str) -> float:
    words = _tokens(text)
    return sum(1 for w in words if len(w) <= 3) / max(len(words), 1)


def feature_commas_per_sentence(text: str) -> float:
    return text.count(",") / max(len(_sentences(text)), 1)


def feature_question_ratio(text: str) -> float:
    n = max(text.count("?") + text.count("!") + text.count("."), 1)
    return text.count("?") / n


def feature_exclamation_ratio(text: str) -> float:
    n = max(text.count("?") + text.count("!") + text.count("."), 1)
    return text.count("!") / n


def feature_punctuation_density(text: str) -> float:
    return sum(1 for c in text if c in _PUNCT) / max(len(text), 1)


def feature_avg_paragraph_length(text: str) -> float:
    paras = [p for p in text.split("\n\n") if p.strip()]
    if not paras:
        return len(_tokens(text))
    return sum(len(p.split()) for p in paras) / len(paras)


def feature_avg_sentence_length(text: str) -> float:
    sents = _sentences(text)
    if not sents:
        return 0.0
    return sum(len(_tokens(s)) for s in sents) / len(sents)


def feature_sentence_length_std(text: str) -> float:
    sents = _sentences(text)
    if len(sents) < 2:
        return 0.0
    lens = [len(_tokens(s)) for s in sents]
    avg = sum(lens) / len(lens)
    var = sum((x - avg) ** 2 for x in lens) / (len(lens) - 1)
    return math.sqrt(var)


def feature_complex_sentence_ratio(text: str) -> float:
    sents = _sentences(text)
    if not sents:
        return 0.0
    return sum(1 for s in sents if len(_tokens(s)) > 30) / len(sents)


def feature_pos_distribution(texts: list[str], language: str) -> np.ndarray:
    out = np.zeros((len(texts), len(_UD_TAGS)), dtype=np.float32)
    tag_idx = {t: i for i, t in enumerate(_UD_TAGS)}
    for i, tags in enumerate(_pos_tags_batch(texts, language)):
        if not tags:
            continue
        cnt = Counter(tags)
        total = len(tags)
        for tag, idx in tag_idx.items():
            out[i, idx] = cnt.get(tag, 0) / total
    return out


def feature_syntactic_depth(texts: list[str], language: str) -> np.ndarray:
    out = np.zeros((len(texts), 2), dtype=np.float32)
    nlp = _load_spacy(language)
    if nlp == "fallback":
        return out
    for i, text in enumerate(texts):
        try:
            doc = nlp(text[:50_000])
            depths, distances = [], []
            for tok in doc:
                if tok.is_space:
                    continue
                d, t = 0, tok
                while t.head is not t and d < 50:
                    d += 1
                    t = t.head
                depths.append(d)
                distances.append(abs(tok.i - tok.head.i))
            if depths:
                out[i] = [np.mean(depths), np.mean(distances)]
        except Exception:
            continue
    return out


def _pos_tags_batch(texts: list[str], language: str) -> list[list[str]]:
    nlp = _load_spacy(language)
    if nlp == "fallback":
        return [_pos_fallback(t) for t in texts]
    out = []
    for text in texts:
        try:
            doc = nlp(text[:50_000])
            out.append([t.pos_ for t in doc if not t.is_space])
        except Exception:
            out.append(_pos_fallback(text))
    return out


def _pos_fallback(text: str) -> list[str]:
    tags = []
    for t in re.findall(r"[A-Za-zА-Яа-яЁё]+|\d+|[.,!?;:]", text):
        if t[0].isdigit():
            tags.append("NUM")
        elif t in ".,!?;:":
            tags.append("PUNCT")
        else:
            tags.append("X")
    return tags


def _identity(x):
    return x


class FeatureExtractor:
    def __init__(self, language: str = "both") -> None:
        self.language = language
        self._func_words = _resolve_func_words(language)
        self._word_vec = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=1500,
            sublinear_tf=True,
            min_df=2,
        )
        self._char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=1000,
            sublinear_tf=True,
            min_df=2,
        )
        self._pos_vec = TfidfVectorizer(
            analyzer="word",
            tokenizer=str.split,
            preprocessor=_identity,
            token_pattern=None,
            ngram_range=(2, 3),
            max_features=500,
            sublinear_tf=True,
            min_df=2,
            lowercase=False,
        )
        self._scalar_mean: np.ndarray | None = None
        self._scalar_std: np.ndarray | None = None

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self._word_vec.fit(texts)
        self._char_vec.fit(texts)
        pos_seqs = [" ".join(tags) for tags in _pos_tags_batch(texts, self.language)]
        self._pos_vec.fit(pos_seqs)

        scalars = self._scalar_matrix(texts)
        self._scalar_mean = scalars.mean(axis=0)
        self._scalar_std = scalars.std(axis=0) + 1e-8

        return self._assemble(texts, pos_seqs, scalars)

    def transform(self, texts: list[str]) -> np.ndarray:
        pos_seqs = [" ".join(tags) for tags in _pos_tags_batch(texts, self.language)]
        scalars = self._scalar_matrix(texts)
        return self._assemble(texts, pos_seqs, scalars)

    def _assemble(
        self, texts: list[str], pos_seqs: list[str], scalars: np.ndarray
    ) -> np.ndarray:
        X_word = self._word_vec.transform(texts)
        X_char = self._char_vec.transform(texts)
        X_pos_ng = self._pos_vec.transform(pos_seqs)
        X_func = feature_function_words(texts, self._func_words)
        X_pos_dist = feature_pos_distribution(texts, self.language)
        X_synt_depth = feature_syntactic_depth(texts, self.language)
        scalars_norm = (scalars - self._scalar_mean) / self._scalar_std
        X = hstack(
            [
                X_word,
                X_char,
                X_pos_ng,
                csr_matrix(X_func),
                csr_matrix(X_pos_dist),
                csr_matrix(X_synt_depth),
                csr_matrix(scalars_norm),
            ]
        ).toarray()
        return X.astype(np.float32)

    def _scalar_matrix(self, texts: list[str]) -> np.ndarray:
        """Все скалярные признаки на текст — одной матрицей (N, K)."""
        fns = [
            feature_hapax_ratio,
            feature_yules_k,
            feature_ttr,
            feature_avg_word_length,
            feature_long_word_ratio,
            feature_short_word_ratio,
            feature_commas_per_sentence,
            feature_question_ratio,
            feature_exclamation_ratio,
            feature_punctuation_density,
            feature_avg_paragraph_length,
            feature_avg_sentence_length,
            feature_sentence_length_std,
            feature_complex_sentence_ratio,
        ]
        return np.array(
            [[fn(t) for fn in fns] for t in texts],
            dtype=np.float32,
        )


def _resolve_func_words(language: str) -> list[str]:
    if language == "ru":
        return RUSSIAN_FUNCTION_WORDS
    if language == "en":
        return ENGLISH_FUNCTION_WORDS
    return list(set(ENGLISH_FUNCTION_WORDS + RUSSIAN_FUNCTION_WORDS))
