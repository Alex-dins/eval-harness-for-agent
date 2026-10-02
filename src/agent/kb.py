"""A tiny, dependency-free knowledge base over local Markdown articles.

Search is deliberately simple and deterministic (TF-IDF-ish keyword scoring) so
that eval runs are reproducible and don't depend on an embedding model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

# Repo-root-relative default corpus location: <repo>/datasets/kb
DEFAULT_KB_DIR = Path(__file__).resolve().parents[2] / "datasets" / "kb"

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    body: str

    def snippet(self, max_chars: int = 240) -> str:
        """A short, single-line preview of the article body."""
        flat = " ".join(self.body.split())
        return flat if len(flat) <= max_chars else flat[: max_chars - 1] + "…"


@dataclass(frozen=True)
class SearchHit:
    id: str
    title: str
    snippet: str
    score: float


class KnowledgeBase:
    """Loads Markdown articles from a directory and ranks them by keyword overlap."""

    def __init__(self, articles: list[Article]):
        self._articles: dict[str, Article] = {a.id: a for a in articles}

    @classmethod
    def from_dir(cls, kb_dir: Path | str = DEFAULT_KB_DIR) -> "KnowledgeBase":
        kb_dir = Path(kb_dir)
        if not kb_dir.is_dir():
            raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")
        articles = [
            _load_article(path) for path in sorted(kb_dir.glob("*.md"))
        ]
        if not articles:
            raise ValueError(f"No .md articles found in {kb_dir}")
        return cls(articles)

    @property
    def article_ids(self) -> list[str]:
        return list(self._articles)

    def get(self, article_id: str) -> Article | None:
        return self._articles.get(article_id)

    @cached_property
    def _doc_freq(self) -> dict[str, int]:
        """Number of articles each term appears in (for IDF weighting)."""
        df: dict[str, int] = {}
        for article in self._articles.values():
            for term in set(_tokenize(article.title + " " + article.body)):
                df[term] = df.get(term, 0) + 1
        return df

    def _score(self, query_terms: list[str], article: Article) -> float:
        n_docs = len(self._articles)
        tokens = _tokenize(article.title + " " + article.body)
        if not tokens:
            return 0.0
        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            idf = math.log((n_docs + 1) / (self._doc_freq.get(term, 0) + 1)) + 1.0
            score += (tf / len(tokens)) * idf
        return score

    def search(self, query: str, k: int = 3) -> list[SearchHit]:
        query_terms = _tokenize(query)
        scored = [
            (self._score(query_terms, a), a) for a in self._articles.values()
        ]
        scored = [(s, a) for s, a in scored if s > 0]
        # Sort by score desc, then id for a stable, deterministic order.
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [
            SearchHit(id=a.id, title=a.title, snippet=a.snippet(), score=round(s, 4))
            for s, a in scored[:k]
        ]


def _load_article(path: Path) -> Article:
    text = path.read_text(encoding="utf-8")
    title = path.stem
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    return Article(id=path.stem, title=title, body=text)
