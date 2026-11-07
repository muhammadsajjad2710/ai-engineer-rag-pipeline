
from __future__ import annotations
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@dataclass
class Hit:
    doc_id: str
    text: str
    score: float
    tenant: str = "public"
    cite: str = ""

class Store:
    def __init__(self):
        self.docs: list[dict] = []
        self.vec = TfidfVectorizer()
        self.matrix = None

    def add(self, doc_id: str, text: str, tenant: str = "public", extra: str = ""):
        self.docs.append({"id": doc_id, "text": text, "tenant": tenant, "extra": extra})
        self._fit()

    def _fit(self):
        if not self.docs:
            self.matrix = None
            return
        self.matrix = self.vec.fit_transform([d["text"] + " " + d.get("extra", "") for d in self.docs])

    def search(self, query: str, tenant: str | None = None, k: int = 3, allow: set[str] | None = None) -> list[Hit]:
        if not self.docs:
            return []
        q = self.vec.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(-scores)
        hits = []
        for i in order:
            d = self.docs[i]
            if tenant is not None and d["tenant"] != tenant:
                continue
            if allow is not None and d["id"] not in allow:
                continue
            hits.append(Hit(d["id"], d["text"], float(scores[i]), d["tenant"], d["id"]))
            if len(hits) >= k:
                break
        return hits

def answer_with_citations(hits: list[Hit], query: str) -> dict:
    if not hits:
        return {"answer": "No evidence found.", "citations": []}
    snippets = " ".join(h.text for h in hits[:3])
    return {"answer": snippets[:400], "citations": [h.cite for h in hits[:3]], "query": query}

