"""Study-note generation from a transcript, using a local model via ollama.

Shape: map over ~5-minute windows, merge deterministically in Python, then one
short reduce pass for structure. The model never re-emits the glossary, so a
sloppy reduce cannot silently drop half the lecture.

Two things are deliberately NOT trusted to the model:
  * timestamps -- nothing in the prompt tells it what time it is, so any
    timestamp it produced would be invented. It cites sentence ids instead and
    Python owns every millisecond.
  * free-form citation ids -- `cites` is constrained to a JSON-Schema enum of
    the sentence ids actually present in that chunk, so a hallucinated id is
    impossible at the token level rather than being cleaned up afterwards.

If ollama is unavailable or the model keeps producing junk, an extractive
fallback runs instead. A recording must never fail because notes failed.

And one thing is checked after the fact, because a 3B model will write notes
about SOMETHING whether or not the passage has content: given a religion
lecture whose audio was mostly lost ("Yeah. Yeah. So,"), it produced
thermodynamics questions; given another, it copied the glycolysis/ATP examples
that used to sit in MAP_SYSTEM. So:
  * a transcript or chunk with too few distinctive words never reaches the
    model at all;
  * no prompt contains subject-matter examples the model could copy;
  * every bullet, term, quiz item and the summary must be made mostly of words
    the lecture actually used (anchor.groundedness), or it is dropped.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import urllib.error
import urllib.request

from . import anchor, config, hardware
from .transcribe import sentence_text

log = logging.getLogger(__name__)

OLLAMA_URL = "http://127.0.0.1:11434"
CHARS_PER_TOKEN = 3.6          # English ASR prose
CHUNK_TOKENS = 1800
OVERLAP_SENTS = 2
MAX_ENUM = 120                 # cap the enum so the grammar stays small

# Below these, there is nothing for a model to summarise and it will invent.
# Counted as distinct content words (anchor.content_tokens), not raw words:
# "Yeah. Yeah. Yeah." is three words and zero content.
MIN_LECTURE_CONTENT = 25
MIN_CHUNK_CONTENT = 12
# Measured on this machine's lectures (anchor.groundedness): real notes scored
# bullets >= 0.41, quiz >= 0.38, summaries >= 0.50; notes invented from the
# prompt or the model's training scored 0.0-0.33.
BULLET_MIN_GROUNDED = 0.34
QUIZ_MIN_GROUNDED = 0.34
SUMMARY_MIN_GROUNDED = 0.40
TERM_MIN_GROUNDED = 0.5        # the term itself must be something they said


# Weak-machine guards. qwen2.5:3b needs ~2.3 GB resident (see config), and on a
# CPU a long lecture can take far longer than the lecture itself.
MIN_TOTAL_RAM_BYTES = int(5.5 * 1024**3)   # the model cannot share a smaller machine
MIN_FREE_RAM_BYTES = int(2.5 * 1024**3)    # Linux/Windows only: see generate()
NOTES_BUDGET_S = int(os.environ.get("LECTERN_NOTES_BUDGET_S", "1800"))
BUDGET_CHECK_AFTER = 2                  # chunks measured before projecting


# ------------------------------------------------------------------ ollama
def _post(path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        OLLAMA_URL + path, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def available() -> bool:
    try:
        req = urllib.request.Request(OLLAMA_URL + "/api/tags")
        with urllib.request.urlopen(req, timeout=4) as r:
            tags = json.loads(r.read())
        names = {m.get("name", "") for m in tags.get("models", [])}
        return _tag(config.LLM_MODEL) in {_tag(n) for n in names}
    except Exception:
        return False


def _tag(name: str) -> str:
    """Exact ollama model tag ("qwen3:4b" and "qwen3:14b" are different
    models; a bare name means ":latest")."""
    return name if ":" in name else f"{name}:latest"


def _chat(system: str, user: str, schema: dict, *, timeout: float = 900,
          temperature: float = 0.1, num_predict: int = 900) -> dict:
    """One constrained call. num_ctx is CONSTANT across every call by design:
    changing it forces ollama to evict and reload the runner."""
    res = _post("/api/chat", {
        "model": config.LLM_MODEL,
        "stream": False,
        "format": schema,
        "keep_alive": config.LLM_KEEP_ALIVE,
        "options": {"num_ctx": config.LLM_NUM_CTX,
                    "temperature": temperature,
                    "num_predict": num_predict},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }, timeout=timeout)
    return json.loads(res["message"]["content"])


# ----------------------------------------------------------------- chunking


def chunk(t: dict) -> list[dict]:
    """Split into ~CHUNK_TOKENS windows, snapped to sentence boundaries."""
    sents = t.get("sents") or []
    if not sents:
        return []
    budget = CHUNK_TOKENS * CHARS_PER_TOKEN
    out, cur, size = [], [], 0
    for s in sents:
        txt = sentence_text(t, s)
        if cur and size + len(txt) > budget:
            out.append(cur)
            cur = cur[-OVERLAP_SENTS:] if len(cur) > OVERLAP_SENTS else []
            size = sum(len(sentence_text(t, x)) for x in cur)
        cur.append(s)
        size += len(txt)
    if cur:
        out.append(cur)
    return [{"sents": c, "t0": c[0]["t0"], "t1": c[-1]["t1"]} for c in out if c]


def _numbered(t: dict, sents: list[dict]) -> str:
    return "\n".join(f"[{s['sid']}] {sentence_text(t, s)}" for s in sents)


# ------------------------------------------------------------------ schemas
def _map_schema(sids: list[int]) -> dict:
    ids = sids[:MAX_ENUM]
    cite = {"type": "array", "minItems": 1, "maxItems": 3,
            "description": "Sentence numbers this came from.",
            "items": {"type": "integer", "enum": ids}}
    # Every field is described: without this the model tends to echo the
    # instructions back into `topic` instead of naming the subject.
    return {
        "type": "object",
        "properties": {
            "topic": {"type": "string",
                      "description": "A short heading naming the subject of this "
                                     "passage, in two to five words taken from "
                                     "the passage. Not a sentence."},
            "bullets": {"type": "array", "minItems": 0, "maxItems": 10,
                        "description": "The revision notes for this passage. "
                                       "Empty if the passage has no subject "
                                       "matter.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string",
                                         "description": "One complete sentence of "
                                                        "notes in your own words."},
                                "cites": cite},
                            "required": ["text", "cites"]}},
            "terms": {"type": "array", "maxItems": 8,
                      "description": "Technical terms defined in this passage.",
                      "items": {
                          "type": "object",
                          "properties": {
                              "term": {"type": "string",
                                       "description": "The term alone, no definition."},
                              "definition": {"type": "string",
                                             "description": "One sentence."},
                              "cites": cite},
                          "required": ["term", "definition", "cites"]}},
        },
        "required": ["topic", "bullets"],
    }


_QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string",
                    "description": "Two or three sentences of plain fact."},
        "quiz": {"type": "array", "minItems": 4, "maxItems": 8,
                 "description": "Exam-style questions with short answers.",
                 "items": {
                     "type": "object",
                     "properties": {
                         "q": {"type": "string", "description": "The question."},
                         "a": {"type": "string", "description": "One or two sentences."}},
                     "required": ["q", "a"]}},
    },
    "required": ["summary", "quiz"],
}

MAP_SYSTEM = """You write revision notes for a university student from a lecture \
transcript produced by speech recognition. Technical terms are sometimes \
misspelled by the recogniser; silently correct them when the intended word is \
obvious from context.

Write each note as a plain statement of fact, exactly as it would appear in a \
student's notebook, using the subject matter of THIS transcript only. Never \
write an instruction about note-taking.

  Good: a direct statement of something said, e.g. "<Term> means <meaning>."
  Bad:  an instruction, e.g. "Define <term>."
  Bad:  a remark about the recording, e.g. "Note the repetition of this phrase."
  Bad:  narration, e.g. "The lecturer explains that <fact>."

If the passage is only filler, greetings, fragments or noise, return an empty \
list of bullets. Writing nothing is correct when nothing was taught.

One fact per note. Keep each under about twenty words; split a note carrying two \
facts into two notes. Prefer the lecturer's own numbers, names and formulas.

Record definitions, locations, quantities, the order steps happen in, causes and \
effects, distinctions between similar things, and anything called important, \
examinable, or a common mistake. Ignore greetings and admin unless a deadline is \
given. Never state a fact that is not in the transcript, never invent a time or \
date, and never mention the transcript, the lecture, or the lecturer."""


REDUCE_SYSTEM = """You are given revision notes already extracted from one \
university lecture.

Write two things. First, a two or three sentence summary of what the lecture \
covered, written as plain statements of fact -- do not say "the lecture" or "this \
covers". Second, between four and eight exam-style questions that test whether a \
student understood the material, each with a one or two sentence answer.

Ask about mechanisms, differences, quantities and consequences rather than \
wording. Base every question and answer strictly on the notes given, and do not \
introduce facts that are not there."""


# -------------------------------------------------------------- generation
def generate(t: dict, *, progress_cb=None) -> dict:
    """Full pipeline. Falls back to extractive notes rather than failing."""
    mode = os.environ.get("LECTERN_NOTES", "auto").lower()
    if mode == "extractive":
        return extractive(t, reason="language model turned off")
    if not available():
        log.warning("ollama or %s unavailable; using extractive notes",
                    config.LLM_MODEL)
        return extractive(t, reason="offline")
    if mode != "llm":
        total = hardware.total_ram_bytes()
        eff = hardware.effective_ram_bytes(total)   # Mac terms: Windows needs ~2x
        if eff is not None and eff < MIN_TOTAL_RAM_BYTES:
            log.warning("%d MB RAM in total; skipping the language model",
                        total // 2**20)
            return extractive(t, reason=f"this computer has too little memory "
                                        f"for the language model "
                                        f"({total // 2**30} GB)")
        # "Available" is only trustworthy off macOS. macOS counts compressed
        # and cached pages as used yet frees them on demand: an idle 8 GB Mac
        # routinely reports ~1 GB available and runs the model fine.
        free = None if hardware.pc.IS_MAC else hardware.available_ram_bytes()
        if free is not None and free < MIN_FREE_RAM_BYTES:
            log.warning("only %d MB RAM free; skipping the language model",
                        free // 2**20)
            return extractive(t, reason=f"not enough free memory for the "
                                        f"language model ({free // 2**20} MB free)")
    try:
        return _generate_llm(t, progress_cb=progress_cb)
    except Exception as e:
        log.exception("LLM note generation failed; falling back")
        return extractive(t, reason=f"model error: {e}")
    finally:
        _unload()


def _unload() -> None:
    """Free the model's RAM now, before whisper or the next lecture may run.
    The calls keep it loaded briefly (LLM_KEEP_ALIVE) so one lecture's
    chunks do not reload 2.3 GB from disk each time."""
    try:
        _post("/api/generate", {"model": config.LLM_MODEL, "keep_alive": 0}, timeout=30)
    except Exception:
        log.debug("could not ask ollama to unload %s", config.LLM_MODEL, exc_info=True)


def _content_count(t: dict, sents: list[dict] | None = None) -> int:
    toks: set[str] = set()
    for s in (sents if sents is not None else t.get("sents") or []):
        toks |= anchor.content_tokens(sentence_text(t, s))
    return len(toks)


def _generate_llm(t: dict, *, progress_cb=None) -> dict:
    chunks = chunk(t)
    if not chunks:
        return extractive(t, reason="empty transcript")
    if _content_count(t) < MIN_LECTURE_CONTENT:
        n = len(t.get("words", {}).get("w") or [])
        return _no_notes(f"too little clear speech was recorded to write notes "
                         f"from ({n} words transcribed)")
    m = anchor.Matcher(t)
    vocab = anchor.transcript_vocab(t)
    sections: list[dict] = []
    terms: dict[str, dict] = {}
    started = time.monotonic()
    ran = 0                           # chunks actually sent to the model
    cut_from_ms: int | None = None    # set when the time budget runs out

    for i, ch in enumerate(chunks):
        if progress_cb:
            progress_cb(i, len(chunks))
        if ran >= BUDGET_CHECK_AFTER:
            per = (time.monotonic() - started) / ran
            if per * (len(chunks) - i) + (time.monotonic() - started) > NOTES_BUDGET_S:
                cut_from_ms = ch["t0"]
                log.warning("model too slow here (%.0f s/part); finishing from "
                            "part %d with extractive notes", per, i + 1)
                break
        if _content_count(t, ch["sents"]) < MIN_CHUNK_CONTENT:
            continue                      # nothing taught here; the model would invent
        sids = [s["sid"] for s in ch["sents"]]
        user = (f"Lecture transcript, sentences {sids[0]} to {sids[-1]}:\n\n"
                f"{_numbered(t, ch['sents'])}")
        data = _call_with_repair(MAP_SYSTEM, user, _map_schema(sids), ch, t)
        ran += 1
        if not data:
            continue

        bullets = []
        for b in (data.get("bullets") or []):
            text = (b.get("text") or "").strip()
            if not text or not anchor.content_tokens(text):
                continue                  # "Yeah." is not a note
            g = anchor.groundedness(text, t, vocab)
            if g < BULLET_MIN_GROUNDED:
                log.info("dropped ungrounded note (%.2f): %s", g, text)
                continue
            cites = [c for c in (b.get("cites") or []) if c in sids]
            a = m.anchor(text, cites, ch["t0"], ch["t1"])
            bullets.append({"text": text, "cites": cites, "grounded": round(g, 2),
                            **a.as_dict()})
        if bullets:
            heading = (data.get("topic") or "").strip()
            if (not heading or len(heading.split()) > 8
                    or anchor.groundedness(heading, t, vocab) < BULLET_MIN_GROUNDED):
                heading = f"Part {i+1}"
            sections.append({"heading": heading, "t0": ch["t0"], "bullets": bullets})

        # Deterministic merge: the model never has to re-emit the glossary.
        for term in (data.get("terms") or []):
            name = (term.get("term") or "").strip()
            definition = (term.get("definition") or "").strip()
            if not name or not definition:
                continue
            if (anchor.groundedness(name, t, vocab) < TERM_MIN_GROUNDED
                    or anchor.groundedness(definition, t, vocab) < BULLET_MIN_GROUNDED):
                log.info("dropped ungrounded term: %s", name)
                continue
            key = re.sub(r"^(the|a|an)\s+", "", name.lower()).rstrip("s").strip()
            cites = [c for c in (term.get("cites") or []) if c in sids]
            a = m.anchor(f"{name} {definition}", cites, ch["t0"], ch["t1"])
            prev = terms.get(key)
            if prev is None or len(definition) > len(prev["definition"]):
                terms[key] = {"term": name, "definition": definition,
                              **a.as_dict()}

    if cut_from_ms is not None:
        # Keep what the model wrote; cover the rest verbatim from the transcript.
        tail = [sec for sec in extractive(t, reason="")["sections"]
                if sec["t0"] >= cut_from_ms]
        sections += tail
        return {"schema": 1, "engine": config.LLM_MODEL, "summary": "",
                "sections": sections, "terms": list(terms.values())[:40],
                "quiz": [], "fallback": None,
                "partial_from_ms": cut_from_ms}

    if not sections:
        return extractive(t, reason="model produced no usable notes")

    # The reduce pass writes new prose, so it needs something real to work
    # from. One or two bullets make an outline a 3B model "completes" from its
    # own training.
    if sum(len(s["bullets"]) for s in sections) < 3:
        return {"schema": 1, "engine": config.LLM_MODEL, "summary": "",
                "sections": sections, "terms": list(terms.values())[:40],
                "quiz": [], "fallback": None}

    summary, quiz = "", []
    try:
        outline = "\n".join(
            f"## {s['heading']}\n" + "\n".join(f"- {b['text']}" for b in s["bullets"])
            for s in sections)[:6000]
        red = _chat(REDUCE_SYSTEM, outline, _QUIZ_SCHEMA, num_predict=1100)
        summary = (red.get("summary") or "").strip()
        if summary and anchor.groundedness(summary, t, vocab) < SUMMARY_MIN_GROUNDED:
            log.info("dropped ungrounded summary: %s", summary)
            summary = ""
        quiz, seen_q = [], set()
        for q in (red.get("quiz") or []):
            if not (q.get("q") and q.get("a")):
                continue
            if anchor.groundedness(f"{q['q']} {q['a']}", t, vocab) < QUIZ_MIN_GROUNDED:
                log.info("dropped ungrounded question: %s", q["q"])
                continue
            key = " ".join(anchor.tokenize(q["q"]))
            if key in seen_q:          # small models repeat questions
                continue
            seen_q.add(key)
            quiz.append(q)
        quiz = quiz[:8]
    except Exception:
        log.warning("reduce pass failed; keeping the mapped notes", exc_info=True)

    return {"schema": 1, "engine": config.LLM_MODEL, "summary": summary,
            "sections": sections, "terms": list(terms.values())[:40],
            "quiz": quiz, "fallback": None}


def _call_with_repair(system: str, user: str, schema: dict, ch: dict, t: dict):
    """Retry ladder. Over-context is a distinct failure that needs a SPLIT, not
    a hotter temperature -- ollama returns an explicit error for it."""
    try:
        return _chat(system, user, schema)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:400]
        except Exception:
            pass
        if "context" in body.lower() or e.code == 400:
            half = max(1, len(ch["sents"]) // 2)
            merged = {"topic": "", "bullets": [], "terms": []}
            for part in (ch["sents"][:half], ch["sents"][half:]):
                if not part:
                    continue
                sids = [s["sid"] for s in part]
                try:
                    d = _chat(system, f"Lecture transcript:\n\n{_numbered(t, part)}",
                              _map_schema(sids))
                except Exception:
                    continue
                merged["topic"] = merged["topic"] or (d.get("topic") or "")
                merged["bullets"] += d.get("bullets") or []
                merged["terms"] += d.get("terms") or []
            return merged if merged["bullets"] else None
        log.warning("chunk failed (%s); skipping", e)
        return None
    except json.JSONDecodeError:
        try:
            return _chat(system, user, schema, temperature=0.45)
        except Exception:
            return None
    except Exception as e:
        log.warning("chunk failed (%s); skipping", e)
        return None


def _no_notes(reason: str) -> dict:
    return {"schema": 1, "engine": "none", "summary": "", "sections": [],
            "terms": [], "quiz": [], "fallback": reason}


# ------------------------------------------------------------- extractive
def extractive(t: dict, *, reason: str = "") -> dict:
    """Zero-dependency fallback. Picks real sentences; never paraphrases."""
    sents = t.get("sents") or []
    if not sents:
        return {"schema": 1, "engine": "extractive", "summary": "",
                "sections": [], "terms": [], "quiz": [], "fallback": reason}
    m = anchor.Matcher(t, window_sizes=(1,))
    scored = []
    for i, s in enumerate(sents):
        toks = anchor.tokenize(m.sentence_text(i))
        if len(toks) < 6:
            continue
        score = sum(m.idf.get(tok, 0.0) for tok in set(toks)) / math.sqrt(len(toks))
        scored.append((score, i))
    scored.sort(reverse=True)
    keep = sorted(i for _, i in scored[:max(6, len(sents) // 6)])

    chunks = chunk(t) or [{"t0": 0, "t1": t.get("duration_ms", 0),
                           "sents": sents}]
    sections = []
    for n, ch in enumerate(chunks):
        lo, hi = ch["sents"][0]["sid"], ch["sents"][-1]["sid"]
        bullets = []
        for i in keep:
            s = sents[i]
            if lo <= s["sid"] <= hi:
                bullets.append({"text": m.sentence_text(i), "cites": [s["sid"]],
                                "t_ms": s["t0"], "tier": "precise",
                                "sid": s["sid"], "score": 1.0,
                                "preview": m.sentence_text(i)[:160]})
        if bullets:
            mins = ch["t0"] // 60000
            sections.append({"heading": f"From {mins}:{ch['t0']//1000%60:02d}",
                             "t0": ch["t0"], "bullets": bullets})
    return {"schema": 1, "engine": "extractive", "summary": "",
            "sections": sections, "terms": [], "quiz": [], "fallback": reason}


# ------------------------------------------------------------------ render
def to_markdown(doc: dict, title: str, recovery_note: str | None = None) -> str:
    """notes.md carries its anchors inline, so they survive an index wipe."""
    def stamp(ms: int) -> str:
        return f"[{ms//60000:02d}:{ms//1000%60:02d}](#t={ms/1000:.1f})"

    out = [f"# {title}", ""]
    if recovery_note:
        out += [f"> {recovery_note}", ""]
    if doc.get("fallback") and not doc.get("sections"):
        out += [f"> No study notes for this recording: {doc['fallback']}.", ""]
    elif doc.get("fallback"):
        out += [f"> Notes were generated without the language model "
                f"({doc['fallback']}). These are the most important sentences "
                f"taken verbatim from the transcript.", ""]
    if doc.get("partial_from_ms") is not None:
        ms = doc["partial_from_ms"]
        out += [f"> The language model was too slow on this computer to finish. "
                f"Notes from {ms//60000:02d}:{ms//1000%60:02d} onward are key "
                f"sentences taken verbatim from the transcript.", ""]
    if doc.get("summary"):
        out += [doc["summary"], ""]
    for sec in doc.get("sections") or []:
        out.append(f"## {sec['heading']}")
        for b in sec["bullets"]:
            mark = "" if b.get("tier") == "precise" else " ~"
            out.append(f"- {b['text']} {stamp(b['t_ms'])}{mark}")
        out.append("")
    if doc.get("terms"):
        out += ["## Key terms", ""]
        for term in doc["terms"]:
            out.append(f"- **{term['term']}** — {term['definition']} {stamp(term['t_ms'])}")
        out.append("")
    if doc.get("quiz"):
        out += ["## Check yourself", ""]
        for i, q in enumerate(doc["quiz"], 1):
            out += [f"{i}. {q['q']}", f"   - {q['a']}"]
        out.append("")
    out += ["---", "", "`~` marks an approximate timestamp: the note summarises a "
            "stretch of the lecture rather than one sentence."]
    return "\n".join(out) + "\n"
