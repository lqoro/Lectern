"""Notes map calls made during the recording, from the live draft.

The map pass (notes.py) is most of the notes stage after Stop: one model call
per ~1800-token part. While a lecture is recording with live transcription
on, the draft (live.py) already holds most of the parts, so this module maps
the finished ones in the background and caches each result in

  .work/notes-premap.json   {"schema": 1, "entries": {key: entry}}

where key = sha256(version + the part's normalized sentences) and version
covers the model, num_ctx, the map prompt and schema and PROMPT_REV, so a
result made under other settings is never reused. An entry keeps the draft's
sentence ids with their normalized text, the part's time span and the raw
model output.

After Stop, notes.generate() on the real transcript reuses an entry for a part
whose text matches (lookup):
  * exact: the same normalized sentences in the same order; cites map
    position for position (sentence ids differ between draft and transcript);
  * fuzzy: an entry for the same stretch of the timeline sharing at least
    FUZZY_MIN_RATIO of its words with the part. The draft is transcribed per
    60-second segment, so every seam can come out differently (measured in
    tests/test_notes_speed.py: exact matching alone reuses nothing on a
    50-minute lecture). A cite is kept only when the sentence it named has
    exactly one sentence with identical text in the part; any other cite is
    dropped, never moved. The model saw the DRAFT, so a fuzzy hit then keeps
    a bullet or term only if every one of its cites survived (a dropped cite
    means it leans on a sentence whose text changed), every number it uses
    is in this part's final sentences, and it uses no word -- raw tokens,
    however short or common: "4", "3.5", "adp", "not", "never" -- that the
    draft part has but the final part does not, i.e. wording only the draft
    heard (_vet). Words in neither (the model's own paraphrase) are allowed,
    as for a freshly mapped part. A topic failing the wording checks is
    dropped. If fewer than MIN_KEPT_FRACTION of the cached bullets
    survive, the part is a miss and the model maps it. Survivors are still
    checked against the real transcript (groundedness) like any model output.
Everything else is mapped by the model as before.

A pre-map call is one plain _chat with the same system prompt, schema and
options as the pipeline's map call, but over the DRAFT's text and sentence
ids, so it is not literally the question the final pass would ask. It
deliberately skips notes._call_with_repair's retry/split ladder: a part
that fails here is simply mapped again after Stop, where the ladder runs.

Recording always wins:
  * it never takes STAGE_LOCK or NOTES_LOCK; it does not start a call while
    either is held or waited for (another lecture's audio stage or notes);
  * it maps only while live transcription is on and keeping up
    (live.LIVE.premap_state); behind, it waits; switched off, paused or
    stopped, it ends;
  * one worker thread, one model call at a time, only for parts of the draft
    that can no longer change (all but the last);
  * Stop never waits for it: stop() marks the session closed and returns, and
    a call still in flight is discarded when it returns (ollama finishes that
    one generation; it cannot be interrupted from here);
  * any failure ends pre-mapping for that lecture with a log line (no lecture
    text) and the user never sees it; the normal post-Stop path maps
    whatever is not cached;
  * when pre-mapping is dropped while the recording goes on (live switched
    itself off, a failure, a pause) and it was the last pre-mapping session,
    the model is unloaded (notes._unload) from a short thread if any
    session made a call since the last unload, so it does not sit in RAM
    through the rest of the recording. Stop does NOT unload -- neither
    stop() nor the worker reading live's "stop" state while the pipeline
    finishes the live draft: the notes after Stop need the model within
    seconds, and they unload it when they are done.
The cache is disposable: recovery never reads it, live.discard() deletes it
with the draft, and the trash purge may delete it.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field

from . import anchor, config, live, notes, storage
from .storage import Lecture
from .transcribe import sentence_text

log = logging.getLogger(__name__)

ENV = "LECTERN_PREMAP"          # "0" turns pre-mapping off
PROMPT_REV = 1                  # bump when a map call's options change
CACHE_REV = 3                   # bump when what an entry means or how it is vetted changes
POLL_S = 5.0                    # how often a recording is looked at
# One map call during recording, at most. A call is ~2k prompt tokens and at
# most num_predict (900) out: well under a minute for qwen2.5:3b on a machine
# with notes_share_ram. Slower than 2 minutes, the machine is too busy to
# pre-map beside a recording; the part is mapped after Stop instead.
CALL_TIMEOUT_S = 120
FUZZY_MIN_RATIO = 0.9           # words shared with the part, for a fuzzy reuse
FUZZY_MIN_OVERLAP = 0.5         # of the part's time span the entry must cover
MIN_KEPT_FRACTION = 0.5         # of a fuzzy hit's bullets that must survive vetting
_RAW = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")      # every word, decimals whole
_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_NUMBER_WORDS = frozenset("""zero one two three four five six seven eight nine ten
eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty
thirty forty fifty sixty seventy eighty ninety hundred thousand million billion
trillion half third quarter""".split())


# ---------------------------------------------------------------- keys
def version() -> str:
    """Everything besides the text that decides a map call's answer."""
    parts = [config.LLM_MODEL, config.LLM_NUM_CTX, PROMPT_REV, CACHE_REV,
             notes.MAP_SYSTEM, notes._map_schema([1])]
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _sents(t: dict, ch: dict) -> list[list]:
    """[[sid, normalized text], ...] of one part."""
    return [[s["sid"], _norm(sentence_text(t, s))] for s in ch["sents"]]


def _key(ver: str, sents: list[list]) -> str:
    body = "\n".join(text for _sid, text in sents)
    return hashlib.sha256(f"{ver}\0{body}".encode()).hexdigest()


# --------------------------------------------------------------- cache
def load(lec: Lecture) -> dict:
    """The cached entries, or {} (missing, unreadable, or another shape)."""
    try:
        raw = json.loads((lec.work / live.PREMAP_NAME).read_text())
    except (OSError, ValueError):
        return {}
    entries = raw.get("entries") if isinstance(raw, dict) else None
    return dict(entries) if isinstance(entries, dict) else {}


def _save(lec: Lecture, entries: dict) -> None:
    """Never creates .work: a lecture moved away must not reappear."""
    storage.atomic_write_json(lec.work / live.PREMAP_NAME,
                              {"schema": 1, "entries": entries}, mkdirs=False)


def map_next(draft: dict, entries: dict, skipped: set, *,
             timeout: float = CALL_TIMEOUT_S, before_call=None) -> tuple[str, dict] | None:
    """Map the first stable part of the draft not cached yet. Returns
    (key, entry), or None when there is nothing left to map. Parts too thin
    for the model (notes.MIN_CHUNK_CONTENT) are added to `skipped`."""
    ver = version()
    for ch in notes.chunk(draft)[:-1]:          # the last part may still grow
        sents = _sents(draft, ch)
        key = _key(ver, sents)
        if key in entries or key in skipped:
            continue
        if notes._content_count(draft, ch["sents"]) < notes.MIN_CHUNK_CONTENT:
            skipped.add(key)
            continue
        user, sids = notes.map_prompt(draft, ch)
        if before_call is not None:
            before_call()
        data = notes._chat(notes.MAP_SYSTEM, user, notes._map_schema(sids),
                           timeout=timeout)
        if not isinstance(data, dict):
            raise ValueError("map result is not an object")
        return key, {"ver": ver, "t0": ch["t0"], "t1": ch["t1"],
                     "sents": sents, "data": data}
    return None


# -------------------------------------------------------------- lookup
def lookup(entries: dict, t: dict, ch: dict, used: set):
    """A reusable map result for part `ch` of the real transcript `t`:
    (data with cites re-grounded to t's sentence ids, "exact" | "fuzzy",
    number of cites dropped), or None. `used` (keys) stops one entry
    answering two parts; the chosen key is added to it."""
    ver = version()
    final = _sents(t, ch)
    key = _key(ver, final)
    e = entries.get(key) if key not in used else None
    kind = "exact"
    if not _valid(e, ver):
        key, e = _fuzzy(entries, ver, final, ch, used)
        kind = "fuzzy"
    if e is None:
        return None
    data, dropped, whole = _reground(e, final, positional=(kind == "exact"))
    if kind == "fuzzy":
        data = _vet(data, final, e["sents"], whole)
        if data is None:
            return None                 # too little survives: the model maps it
    used.add(key)
    return data, kind, dropped


def _valid(e, ver: str) -> bool:
    return (isinstance(e, dict) and e.get("ver") == ver
            and isinstance(e.get("data"), dict) and isinstance(e.get("sents"), list))


def _fuzzy(entries: dict, ver: str, final: list[list], ch: dict, used: set):
    span = max(1, ch["t1"] - ch["t0"])
    words = " ".join(text for _sid, text in final).split()
    best, best_ratio = (None, None), FUZZY_MIN_RATIO
    for key, e in entries.items():
        if key in used or not _valid(e, ver):
            continue
        overlap = min(ch["t1"], e.get("t1", 0)) - max(ch["t0"], e.get("t0", 0))
        if overlap < FUZZY_MIN_OVERLAP * span:
            continue
        other = " ".join(text for _sid, text in e["sents"]).split()
        sm = difflib.SequenceMatcher(None, words, other, autojunk=False)
        if sm.real_quick_ratio() < best_ratio or sm.quick_ratio() < best_ratio:
            continue
        r = sm.ratio()
        if r >= best_ratio:
            best, best_ratio = (key, e), r
    return best


def _reground(e: dict, final: list[list], *,
              positional: bool) -> tuple[dict, int, dict]:
    """A copy of the entry's model output with every cite naming the final
    transcript's sentence that says the same thing, or dropped. Also returns,
    per bullet and term, whether ALL its cites survived."""
    if positional:
        remap = {d[0]: f[0] for d, f in zip(e["sents"], final) if d[1] == f[1]}
    else:
        by_text: dict[str, list] = {}
        for sid, text in final:
            by_text.setdefault(text, []).append(sid)
        remap = {sid: by_text[text][0] for sid, text in e["sents"]
                 if len(by_text.get(text, ())) == 1}
    dropped = 0

    def cites(item: dict) -> list:
        nonlocal dropped
        out = []
        for c in item.get("cites") or []:
            new = remap.get(c)
            if new is None:
                dropped += 1
            elif new not in out:
                out.append(new)
        return out

    def whole(item: dict) -> bool:
        return all(c in remap for c in item.get("cites") or [])

    data = e["data"]
    bullets = [b for b in (data.get("bullets") or []) if isinstance(b, dict)]
    terms = [x for x in (data.get("terms") or []) if isinstance(x, dict)]
    out = {"topic": data.get("topic") or "",
           "bullets": [{**b, "cites": cites(b)} for b in bullets],
           "terms": [{**x, "cites": cites(x)} for x in terms]}
    return out, dropped, {"bullets": [whole(b) for b in bullets],
                          "terms": [whole(x) for x in terms]}


def _raw(text: str) -> set[str]:
    return set(_RAW.findall((text or "").lower()))


def _numerals(tokens: set[str]) -> set[str]:
    return {t for t in tokens if _NUMBER.fullmatch(t) or t in _NUMBER_WORDS}


def _vet(data: dict, final: list[list], draft: list[list], whole: dict) -> dict | None:
    """A fuzzy hit's notes, keeping only what this final part supports. A
    bullet or term is kept when it has at least one cite and EVERY original
    cite survived re-grounding (whole), every number it uses is in the
    part's final sentences, and it uses no raw token that the draft part's
    sentences have and the final part's do not. Raw tokens: no length or
    common-word filter, so "4", "3.5", "adp", "pH", "not", "never" count.
    None when fewer than MIN_KEPT_FRACTION of the bullets survive.

    Residual limit, stated plainly: this compares word SETS per part. A
    negation or number the draft heard wrongly that also occurs elsewhere in
    the same final part ("not" said two sentences later), combined with the
    model citing a sentence both transcripts heard alike, cannot be told
    apart from a correct note. That is the same exposure as the model
    misreading a freshly mapped part, which nothing here checks either."""
    final_raw = set().union(*(_raw(text) for _sid, text in final))
    draft_raw = set().union(*(_raw(text) for _sid, text in draft))
    nums = _numerals(final_raw)
    draft_only = draft_raw - final_raw

    def said(text: str) -> bool:
        toks = _raw(text)
        return _numerals(toks) <= nums and not (toks & draft_only)

    def keep(item: dict, is_whole: bool, text: str) -> bool:
        return bool(item["cites"]) and is_whole and said(text)

    bullets = [b for b, w in zip(data["bullets"], whole["bullets"])
               if keep(b, w, b.get("text") or "")]
    cached = len(data["bullets"])
    if cached and (not bullets or len(bullets) < MIN_KEPT_FRACTION * cached):
        return None
    terms = [x for x, w in zip(data["terms"], whole["terms"])
             if keep(x, w, f"{x.get('term') or ''} {x.get('definition') or ''}")]
    topic = data["topic"] if said(data["topic"]) else ""
    return {"topic": topic, "bullets": bullets, "terms": terms}


# -------------------------------------------------------------- worker
def enabled() -> bool:
    """On by default where notes may share the machine with live work."""
    from .pipeline import notes_share_ram
    if os.environ.get(ENV, "1") == "0":
        return False
    if os.environ.get("LECTERN_NOTES", "auto").lower() == "extractive":
        return False
    return notes_share_ram()


def _llm_busy() -> bool:
    """Another lecture's audio stage or notes holds or awaits its lock."""
    from .pipeline import NOTES_LOCK, STAGE_LOCK
    return STAGE_LOCK.busy() or NOTES_LOCK.busy()


def _unload_if_idle() -> None:
    """Unless another lecture's notes have the model (they unload it when
    they finish)."""
    try:
        if not _llm_busy():
            notes._unload()
    except Exception:
        log.debug("could not unload the notes model", exc_info=True)


@dataclass(eq=False)
class _Session:
    lecture_id: str
    closed: bool = False
    checked: bool = False           # enabled() and the model were checked
    entries: dict | None = None     # loaded from .work on first use
    skipped: set = field(default_factory=set)
    epoch: int = 0                  # bumped by every begin(): a resume
    lock: threading.Lock = field(default_factory=threading.Lock)


class PreMapper:
    """One worker thread for all recordings. begin() is memory only."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, _Session] = {}
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._called = False         # a model call was made since the last unload

    def begin(self, lecture_id: str) -> None:
        """A recording (re)started. Memory only; never raises into the caller."""
        with self._lock:
            s = self._sessions.get(lecture_id)
            if s is None or s.closed:
                self._sessions[lecture_id] = _Session(lecture_id)
            else:
                s.epoch += 1            # a pending drop of the old state is void
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="notes-premap",
                                                daemon=True)
                self._thread.start()
        self._wake.set()

    def stop(self, lecture_id: str) -> None:
        """Stop pre-mapping this lecture now (Stop). Never waits for a model
        call: once this returns, nothing more is written to its cache. The
        model stays loaded for the notes that follow."""
        with self._lock:
            s = self._sessions.pop(lecture_id, None)
            if not self._sessions:
                self._called = False    # the notes after Stop take the model over
        if s is not None:
            with s.lock:
                s.closed = True     # no unload: the notes after Stop use the model
        self._wake.set()

    def _drop(self, s: _Session, epoch: int | None = None, *,
              unload: bool = True) -> None:
        """End a session from the worker. With `epoch`, only if no begin()
        (a resume) came in since that state was read. unload=False when the
        recording is being stopped (the notes after Stop use the model)."""
        with self._lock:
            if epoch is not None and s.epoch != epoch:
                return
            if self._sessions.get(s.lecture_id) is s:
                del self._sessions[s.lecture_id]
            free = not self._sessions and self._called
            if not self._sessions:
                self._called = False
        with s.lock:
            s.closed = True
        if free and unload:
            # Free the model's RAM, as the notes stage does when it finishes.
            # A thread: the worker never waits on ollama for this.
            threading.Thread(target=_unload_if_idle, name="notes-premap-unload",
                             daemon=True).start()

    def _mark_called(self) -> None:
        with self._lock:
            self._called = True

    def _run(self) -> None:
        while True:
            self._wake.clear()
            with self._lock:
                todo = list(self._sessions.values())
                if not todo:
                    self._thread = None
                    return
            for s in todo:
                try:
                    self._step(s)
                except Exception as e:
                    log.warning("pre-mapping notes for %s stopped (%s); the notes "
                                "are written after Stop as usual",
                                s.lecture_id, type(e).__name__)
                    self._drop(s)
            self._wake.wait(POLL_S)

    def _step(self, s: _Session) -> None:
        """At most one model call for one recording."""
        if s.closed:
            return
        if not s.checked and not enabled():
            self._drop(s)
            return
        epoch = s.epoch
        state = live.LIVE.premap_state(s.lecture_id)
        if state in ("end", "stop"):
            self._drop(s, epoch, unload=(state == "end"))
            return
        if state != "go" or _llm_busy():
            return
        lec = Lecture.load(s.lecture_id)
        try:
            draft = json.loads(lec.draft_path.read_text())
        except (OSError, ValueError):
            return                              # no draft yet
        if len(notes.chunk(draft)) < 2:
            return                              # nothing stable to map yet
        if not s.checked:
            s.checked = True
            if not notes.available():
                log.info("pre-mapping notes for %s is off: the model is not available",
                         s.lecture_id)
                self._drop(s)
                return
        if s.entries is None:
            s.entries = load(lec)
        t0 = time.monotonic()
        got = map_next(draft, s.entries, s.skipped, before_call=self._mark_called)
        if got is None:
            return
        with s.lock:
            if s.closed:
                return                          # stopped meanwhile: discard
            s.entries = {**s.entries, got[0]: got[1]}
            _save(lec, s.entries)
        log.info("notes for %s: pre-mapped part %d during the recording in %.1f s",
                 s.lecture_id, len(s.entries), time.monotonic() - t0)


PREMAP = PreMapper()
