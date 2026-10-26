"""Grounding generated notes back to the moment they came from.

A note bullet is a paraphrase -- "Substituents prefer equatorial positions" may
appear nowhere verbatim in the transcript -- so anchoring cannot be string
search. Three tiers, in descending precision:

  exact    a word the user clicked in the transcript. Cannot be wrong.
  precise  the model cited a sentence id AND a TF-IDF check agrees.
  region   everything else: the bullet inherits its chunk's time range. Coarse
           (a few minutes) but structurally incapable of being wrong.

The UI must show which tier an anchor is. An anchor that quietly lies is worse
than one that admits it is approximate.

Implementation note: TF-IDF cosine over an inverted index, pure stdlib. Measured
alternative `difflib.SequenceMatcher` is ~2,600x slower and compares characters,
which is exactly the wrong similarity for a paraphrase.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")
# A term appearing in more than this fraction of documents carries no signal and
# its postings list is as long as the corpus, so it dominates query cost.
_DF_CEILING = 0.30
_MIN_LEN = 2

PRECISE_MIN = 0.28          # cosine above which a cited sentence is trusted
REPAIR_MIN = 0.42           # cosine above which a match can override a bad cite


@dataclass
class Anchor:
    t_ms: int
    tier: str                # "precise" | "region"
    sid: int | None
    score: float
    preview: str = ""

    def as_dict(self) -> dict:
        return {"t_ms": self.t_ms, "tier": self.tier, "sid": self.sid,
                "score": round(self.score, 3), "preview": self.preview}


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall((text or "").lower()) if len(t) >= _MIN_LEN]


class Matcher:
    """TF-IDF cosine over sentence windows of one lecture."""

    def __init__(self, transcript: dict, window_sizes: tuple[int, ...] = (1, 2, 3)):
        self.t = transcript
        self.sents = transcript.get("sents") or []
        self.words = transcript.get("words", {}).get("w") or []
        self.docs: list[tuple[int, int]] = []          # (first sid idx, n sents)
        for n in window_sizes:
            for i in range(0, max(0, len(self.sents) - n + 1)):
                self.docs.append((i, n))
        self._build()

    # -- text access
    def sentence_text(self, idx: int) -> str:
        s = self.sents[idx]
        return " ".join(self.words[s["w0"]: s["w0"] + s["wn"]])

    def _doc_text(self, d: tuple[int, int]) -> str:
        i, n = d
        return " ".join(self.sentence_text(k) for k in range(i, min(i + n, len(self.sents))))

    # -- index
    def _build(self) -> None:
        N = len(self.docs)
        self.N = N
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.norms: list[float] = [0.0] * N
        if not N:
            self.idf = {}
            return
        tfs: list[dict[str, int]] = []
        for d in self.docs:
            counts: dict[str, int] = {}
            for tok in tokenize(self._doc_text(d)):
                counts[tok] = counts.get(tok, 0) + 1
            tfs.append(counts)

        # Document frequency is counted over SENTENCES, not windows. The windows
        # overlap, so a term in one sentence appears in up to six of them, which
        # would inflate its df ~6x and prune ordinary vocabulary. The ceiling is
        # also only meaningful once the corpus is big enough to have stopword
        # statistics at all -- on a three-sentence lecture it would prune
        # everything and the matcher would return nothing.
        ns = max(1, len(self.sents))
        df: dict[str, int] = {}
        for i in range(len(self.sents)):
            for tok in set(tokenize(self.sentence_text(i))):
                df[tok] = df.get(tok, 0) + 1
        ceiling = ns if ns < 30 else max(3, int(_DF_CEILING * ns))
        self.idf = {t: math.log(1 + ns / c) for t, c in df.items() if c <= ceiling}
        for j, counts in enumerate(tfs):
            sq = 0.0
            for tok, c in counts.items():
                idf = self.idf.get(tok)
                if idf is None:
                    continue                     # pruned: stopword-like
                w = (1 + math.log(c)) * idf
                sq += w * w
                self.postings.setdefault(tok, []).append((j, c))
            self.norms[j] = math.sqrt(sq) or 1.0

    def search(self, text: str, top: int = 1) -> list[tuple[int, float]]:
        """Return [(doc index, cosine)] best-first."""
        if not self.N:
            return []
        q: dict[str, int] = {}
        for tok in tokenize(text):
            q[tok] = q.get(tok, 0) + 1
        qw: dict[str, float] = {}
        qsq = 0.0
        for tok, c in q.items():
            idf = self.idf.get(tok)
            if idf is None:
                continue
            w = (1 + math.log(c)) * idf
            qw[tok] = w
            qsq += w * w
        if not qw:
            return []
        qnorm = math.sqrt(qsq)
        scores: dict[int, float] = {}
        for tok, w in qw.items():
            for j, c in self.postings.get(tok, ()):
                scores[j] = scores.get(j, 0.0) + w * (1 + math.log(c)) * self.idf[tok]
        ranked = sorted(((j, s / (self.norms[j] * qnorm)) for j, s in scores.items()),
                        key=lambda x: -x[1])
        return ranked[:top]

    # -- the actual anchoring decision
    def anchor(self, text: str, cited_sids, region_ms: int,
               region_end_ms: int | None = None) -> Anchor:
        """Decide where a bullet points, and how much to trust it."""
        if cited_sids is None:
            cited_sids = []
        elif isinstance(cited_sids, int):
            cited_sids = [cited_sids]

        best_idx, best_score = (None, 0.0)
        hits = self.search(text, top=1)
        if hits:
            best_idx, best_score = hits[0][0], hits[0][1]

        # 1. Trust the model's citation when the text agrees with it. A bullet
        #    often cites two or three sentences; pick the one it actually
        #    matches, not simply the first, or the preview shows the wrong line.
        cand = []
        for sid in cited_sids:
            si = self._sid_index(sid)
            if si is not None:
                cand.append((self._score_against(text, si), si, sid))
        if cand:
            agree, si, sid = max(cand)
            if agree >= PRECISE_MIN:
                return Anchor(self.sents[si]["t0"], "precise", sid,
                              agree, self.sentence_text(si)[:160])

        # 2. No usable citation: accept a strong independent match instead.
        if best_idx is not None and best_score >= REPAIR_MIN:
            i, _n = self.docs[best_idx]
            if self._in_region(self.sents[i]["t0"], region_ms, region_end_ms):
                return Anchor(self.sents[i]["t0"], "precise", self.sents[i]["sid"],
                              best_score, self.sentence_text(i)[:160])

        # 3. Fall back to the chunk's own time range. Coarse but never wrong.
        return Anchor(region_ms, "region", None, best_score, "")

    def _sid_index(self, sid: int) -> int | None:
        for i, s in enumerate(self.sents):
            if s.get("sid") == sid:
                return i
        return None

    def _score_against(self, text: str, si: int) -> float:
        """Cosine of `text` against the 1-3 sentence windows starting at si."""
        best = 0.0
        for j, d in enumerate(self.docs):
            if d[0] != si:
                continue
            for k, sc in self.search(text, top=8):
                if k == j:
                    best = max(best, sc)
        if best:
            return best
        # search() truncates; score this one document directly as a fallback
        target = " ".join(self.sentence_text(k)
                          for k in range(si, min(si + 2, len(self.sents))))
        return _cosine(tokenize(text), tokenize(target), self.idf)

    @staticmethod
    def _in_region(t_ms: int, start_ms: int, end_ms: int | None) -> bool:
        if end_ms is None:
            return True
        pad = 15000
        return (start_ms - pad) <= t_ms <= (end_ms + pad)


# Words that carry no subject matter. Without this, a note invented from the
# model's own training ("Glycolysis occurs in the cytoplasm and does not require
# oxygen") scored as half-grounded in a religion lecture on "does", "require"
# and "occurs" alone.
_COMMON = frozenset("""
about above after again against also although always among another around
because been before being below between both came cannot could does doing done
down during each either even every example first from further have having here
however important include included includes including into itself just know
known last like made main make makes making many might more most much must
never next often only other others over part same second should show shows
since some still such take than that their them then there therefore these
they thing things this those though three through time together under until
upon used uses using very want well were what when where whether which while
whom whose will with within without would your yours called call calls means
mean different another point points state states idea ideas people person
""".split())


def _stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s", "ly"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)]
    return t


def content_tokens(text: str) -> set[str]:
    """Distinctive words of a text, lightly stemmed so a paraphrase that says
    "argued" still matches a lecture that said "argues"."""
    return {_stem(t) for t in tokenize(text) if len(t) >= 4 and t not in _COMMON}


def transcript_vocab(transcript: dict) -> set[str]:
    vocab: set[str] = set()
    for w in transcript.get("words", {}).get("w") or []:
        vocab.update(_stem(t) for t in tokenize(w))
    return vocab


def groundedness(text: str, transcript: dict, vocab: set[str] | None = None) -> float:
    """Fraction of a note's distinctive words that occur in the lecture.

    A small model sometimes supplies a fact from its own training that the
    lecturer never said -- or, given a near-empty transcript, copies the
    examples out of its own prompt. Such a note is made of vocabulary the
    lecture never used. Legitimate paraphrase introduces some synonyms too, so
    notes.py drops only notes well below what real notes score.
    """
    if vocab is None:
        vocab = transcript_vocab(transcript)
    toks = content_tokens(text)
    if not toks:
        return 1.0
    return sum(1 for t in toks if t in vocab) / len(toks)


def _cosine(a: list[str], b: list[str], idf: dict[str, float]) -> float:
    ca: dict[str, int] = {}
    cb: dict[str, int] = {}
    for t in a:
        ca[t] = ca.get(t, 0) + 1
    for t in b:
        cb[t] = cb.get(t, 0) + 1
    num = 0.0
    na = nb = 0.0
    for t, c in ca.items():
        w = (1 + math.log(c)) * idf.get(t, 0.0)
        na += w * w
        if t in cb:
            num += w * (1 + math.log(cb[t])) * idf.get(t, 0.0)
    for t, c in cb.items():
        w = (1 + math.log(c)) * idf.get(t, 0.0)
        nb += w * w
    if not na or not nb:
        return 0.0
    return num / math.sqrt(na * nb)
