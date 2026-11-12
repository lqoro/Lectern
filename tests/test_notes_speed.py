"""The post-Stop notes stage, faster without weaker safety: per-stage timing,
notes computed beside the archive, and map results computed during the
recording from the live draft and reused after Stop.

A fake model only -- never ollama, whisper or the network.
Run: python -m unittest discover -s tests"""
import importlib, json, logging, os, shutil, sys, tempfile, threading, time, unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GB = 1024 ** 3
SEAM_MS = 60000


# No test here may reach a real ollama, even from a thread that outlives its
# test's mocks: every HTTP request fails at the source.
def _no_network(*a, **k):
    raise AssertionError("a test tried to make a network request")


_GUARD = mock.patch.object(urllib.request, "urlopen", _no_network)


def setUpModule():
    _GUARD.start()


def tearDownModule():
    _GUARD.stop()


def fresh(tmp):
    os.environ["LECTERN_ROOT"] = str(Path(tmp) / "Lectures")
    os.environ["LECTERN_SUPPORT"] = str(Path(tmp) / "support")
    os.environ.setdefault("LECTERN_WARMUP", "0")
    for name in [m for m in sys.modules if m.startswith("lectern")]:
        del sys.modules[name]
    config = importlib.import_module("lectern.config")
    config.ensure_dirs()
    return config


def wait_for(pred, timeout=5.0, what="condition"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


# ------------------------------------------------------------- transcripts
def lecture_words(minutes, seed=0):
    """(words, t0s): a lecture of distinct-ish content words, ~13 sentences a
    minute, every sentence ending in a full stop."""
    words, t0s = [], []
    t, k = 0, seed
    while t < minutes * 60000:
        n = 9 + k % 6
        for j in range(n):
            k += 1
            w = f"topic{k % 97}x{(k * 7) % 53}"
            words.append(w + ("." if j == n - 1 else ""))
            t0s.append(t)
            t += 380
    return words, t0s


def build(words, t0s):
    """The transcript contract (transcribe.normalize's shape) from words."""
    sents, start = [], 0
    for i, w in enumerate(words):
        if w.endswith("."):
            sents.append({"sid": len(sents) + 1, "w0": start, "wn": i - start + 1,
                          "t0": t0s[start], "t1": t0s[i] + 300})
            start = i + 1
    if start < len(words):
        sents.append({"sid": len(sents) + 1, "w0": start, "wn": len(words) - start,
                      "t0": t0s[start], "t1": t0s[-1] + 300})
    return {"schema": 1, "duration_ms": (t0s[-1] + 300) if t0s else 0,
            "language": "en",
            "words": {"w": list(words), "t0": list(t0s), "t1": [x + 300 for x in t0s],
                      "p": [1.0] * len(words)},
            "sents": sents}


def seam_draft(words, t0s):
    """What the live draft looks like beside the full pass: the same speech,
    but at every 60-second segment seam the word cut by the seam comes out
    differently -- dropped, truncated, or closed with a spurious full stop
    (an engine punctuates the end of its audio)."""
    out_w, out_t = [], []
    seam, kind = SEAM_MS, 0
    for w, t in zip(words, t0s):
        if t >= seam:
            seam += SEAM_MS
            kind += 1
            if kind % 3 == 0:
                continue                                   # dropped
            if kind % 3 == 1:
                w = w[:max(1, len(w) // 2)] + ("." if w.endswith(".") else "")
            else:
                w = w if w.endswith(".") else w + "."      # spurious sentence end
        out_w.append(w)
        out_t.append(t)
    return out_w, out_t


class FakeModel:
    """Stands in for notes._chat. A map call returns three notes: the 3rd,
    6th and 9th numbered sentences verbatim, each citing itself. `inject`
    adds a fourth note: that word followed by the 3rd sentence, citing it --
    wording only the transcript this call saw may contain (a mishearing in
    the draft). `paraphrase` appends words that are in no transcript, as a
    model writing "in its own words" does. A reduce returns nothing."""

    PARAPHRASE = "summarised concisely"

    def __init__(self, fail=False, block=None, inject=None, delay=0.0, paraphrase=False):
        self.maps, self.reduces = [], 0
        self.fail, self.block, self.inject, self.delay = fail, block, inject, delay
        self.paraphrase = paraphrase
        self.lock = threading.Lock()
        self.started = threading.Event()
        self.returned = threading.Event()

    def __call__(self, system, user, schema, **kw):
        self.started.set()
        try:
            return self._answer(user, schema)
        finally:
            self.returned.set()

    def _answer(self, user, schema):
        if self.block is not None:
            self.block.wait(10)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise TimeoutError("model timed out")
        if "bullets" not in schema.get("properties", {}):
            with self.lock:
                self.reduces += 1
            return {"summary": "", "quiz": []}
        lines = [l for l in user.splitlines() if l.startswith("[")]
        picked = []
        for i in (2, 5, 8):
            line = lines[min(i, len(lines) - 1)]
            picked.append((int(line[1:line.index("]")]), line.split("] ", 1)[1]))
        with self.lock:
            self.maps.append(picked[0][0])
        tail = f" {self.PARAPHRASE}" if self.paraphrase else ""
        bullets = [{"text": text + tail, "cites": [sid]} for sid, text in picked]
        if self.inject:
            bullets.append({"text": f"{self.inject} {picked[0][1]}", "cites": [picked[0][0]]})
        return {"topic": "", "bullets": bullets, "terms": []}


class Base(unittest.TestCase):
    RAM = 32 * GB

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        env = mock.patch.dict(os.environ, {"LECTERN_NOTES": "llm"})
        env.start()
        self.addCleanup(env.stop)
        self.config = fresh(self.tmp)
        self.storage = importlib.import_module("lectern.storage")
        self.Lecture = self.storage.Lecture
        self.notes = importlib.import_module("lectern.notes")
        self.pipeline = importlib.import_module("lectern.pipeline")
        self.premap = importlib.import_module("lectern.premap")
        self.live = importlib.import_module("lectern.live")
        self.recovery = importlib.import_module("lectern.recovery")
        for target, value in ((self.pipeline.hardware, "effective_ram_bytes"),):
            p = mock.patch.object(target, value, lambda *a, **k: self.RAM)
            p.start()
            self.addCleanup(p.stop)
        # Never the network: no ollama, no unload request.
        for name, value in (("_post", lambda *a, **k: {}), ("available", lambda: True)):
            p = mock.patch.object(self.notes, name, value)
            p.start()
            self.addCleanup(p.stop)

    def model(self, **kw):
        fake = FakeModel(**kw)
        p = mock.patch.object(self.notes, "_chat", fake)
        p.start()
        self.addCleanup(p.stop)
        return fake

    def norm(self, text):
        return " ".join(text.lower().split())


# =================================================================== timing
class PipelineHarness(Base):
    def setUp(self):
        super().setUp()
        self.p = self.pipeline.Pipeline()
        self.doc = build(*lecture_words(14))
        self.jobs = []                       # Jobs run directly, outside self.p
        self.addCleanup(self.join_notes)     # runs before the mocks are undone

    def join_notes(self):
        """No notes thread outlives its test (and its mocks)."""
        for job in list(self.p._jobs.values()) + self.jobs:
            run = job.notes_run
            if run is not None:
                run.cancel.set()
                self.assertTrue(run.done.wait(10), "a notes thread did not finish")
        wait_for(lambda: not self.pipeline.NOTES_LOCK.busy(), what="notes slot freed")

    def wait_done(self, *ids, timeout=10):
        end = time.time() + timeout
        while time.time() < end:
            if all((self.p.job(i) or {}).get("done") for i in ids):
                return
            time.sleep(0.01)
        self.fail(f"jobs did not finish: {[self.p.job(i) for i in ids]}")

    def stages(self, lec, *, archive=None, concat=None):
        def transcribe(job, l):
            self.storage.atomic_write_json(l.transcript_path, self.doc)

        def default(job, l):
            time.sleep(0.02)
        return {"_concat": concat or default, "_transcribe": transcribe,
                "_archive": archive or default}


class Timing(PipelineHarness):
    def test_stage_seconds_are_stored_next_to_notes_compute_s(self):
        self.model()
        lec = self.Lecture.create("Timed", state="paused")
        with mock.patch.multiple(self.p, **self.stages(lec)):
            self.p.submit(lec.id)
            self.wait_done(lec.id)
        m = self.Lecture.load(lec.id).manifest()
        self.assertEqual(m["state"], "complete", self.p.job(lec.id))
        for k in ("concat_s", "transcribe_s", "archive_s", "notes_s"):
            self.assertIsInstance(m["timing"][k], (int, float), k)
        self.assertGreaterEqual(m["timing"]["concat_s"], 0.02)
        n = m["notes"]
        self.assertIn("compute_s", n)                      # kept
        self.assertEqual(len(n["map_s"]), len(self.notes.chunk(self.doc)))
        self.assertTrue(all(isinstance(x, (int, float)) for x in n["map_s"]))
        self.assertIsInstance(n["reduce_s"], (int, float))

    def test_one_summary_line_without_lecture_text(self):
        self.model()
        lec = self.Lecture.create("Logged", state="paused")
        with mock.patch.multiple(self.p, **self.stages(lec)), \
                self.assertLogs(self.pipeline.log, logging.INFO) as cm:
            self.p.submit(lec.id)
            self.wait_done(lec.id)
        lines = [r for r in cm.output if "timing" in r]
        self.assertEqual(len(lines), 1, cm.output)
        self.assertIn(lec.id, lines[0])
        for w in self.doc["words"]["w"][:200]:
            self.assertNotIn(w.rstrip("."), lines[0])

    def test_rewrite_notes_records_notes_timing(self):
        self.model()
        lec = self.Lecture.create("Again", state="complete")
        self.storage.atomic_write_json(lec.transcript_path, self.doc)
        self.p.submit_notes(lec.id)
        self.wait_done(lec.id)
        n = self.Lecture.load(lec.id).manifest()["notes"]
        self.assertIn("compute_s", n)
        self.assertEqual(len(n["map_s"]), len(self.notes.chunk(self.doc)))

    def test_generate_reports_per_part_seconds(self):
        self.model()
        stats = {}
        nd = self.notes.generate(self.doc, stats=stats)
        self.assertEqual(nd["engine"], self.config.LLM_MODEL)
        self.assertEqual(len(stats["map_s"]), len(self.notes.chunk(self.doc)))
        self.assertIn("reduce_s", stats)
        self.assertEqual(stats["premap"], {"exact": 0, "fuzzy": 0, "cites_dropped": 0})


# ================================================================= overlap
class Overlap(PipelineHarness):
    def test_notes_are_computed_while_the_archive_runs(self):
        fake = self.model()
        lec = self.Lecture.create("Both", state="paused")
        seen = {}

        def archive(job, l):
            # The model is working on this lecture before the archive is done...
            seen["model_during_archive"] = fake.started.wait(5)
            m = self.Lecture.load(l.id).manifest()
            # ...but nothing about notes is final, and recovery sees no "noting".
            seen["state"] = m.get("state")
            seen["notes_files"] = [p.name for p in (l.notes_path, l.path / "notes.json")
                                   if p.exists()]
        with mock.patch.multiple(self.p, **self.stages(lec, archive=archive)):
            self.p.submit(lec.id)
            self.wait_done(lec.id)
        self.assertTrue(seen["model_during_archive"])
        self.assertNotEqual(seen["state"], "noting")
        self.assertEqual(seen["notes_files"], [])
        m = self.Lecture.load(lec.id).manifest()
        self.assertEqual(m["state"], "complete")
        self.assertTrue((lec.path / "notes.json").is_file())
        self.assertTrue(m["timing"]["overlap"])

    def test_small_machine_keeps_the_serial_order(self):
        self.RAM = 8 * GB
        fake = self.model()
        lec = self.Lecture.create("Serial", state="paused")
        seen = {}

        def archive(job, l):
            time.sleep(0.1)
            seen["model_before_archive_done"] = fake.started.is_set()
        with mock.patch.multiple(self.p, **self.stages(lec, archive=archive)):
            self.p.submit(lec.id)
            self.wait_done(lec.id)
        self.assertFalse(seen["model_before_archive_done"])
        m = self.Lecture.load(lec.id).manifest()
        self.assertEqual(m["state"], "complete")
        self.assertFalse(m["timing"]["overlap"])

    def test_an_archive_failure_surfaces_and_notes_are_never_saved(self):
        gate = threading.Event()
        fake = self.model(block=gate)
        lec = self.Lecture.create("Broken archive", state="paused")

        def archive(job, l):
            fake.started.wait(5)
            raise RuntimeError("archive did not verify")
        with mock.patch.multiple(self.p, **self.stages(lec, archive=archive)):
            self.p.submit(lec.id)
            self.wait_done(lec.id)
            gate.set()
            run = self.p._jobs[lec.id].notes_run
            self.assertTrue(run.done.wait(5))   # it noticed it was cancelled
        job = self.p.job(lec.id)
        self.assertEqual(job["stage"], "failed")
        self.assertIn("archive did not verify", job["error"])
        m = self.Lecture.load(lec.id).manifest()
        self.assertEqual(m["state"], "failed")
        self.assertIn("archive did not verify", m["failure"])
        self.assertFalse((lec.path / "notes.json").exists())
        self.assertFalse(lec.notes_path.exists())
        # Cancelled between parts: the model was not run over the whole lecture.
        self.assertLess(len(fake.maps), len(self.notes.chunk(self.doc)))

    def test_a_notes_failure_surfaces_after_the_archive(self):
        self.model()
        lec = self.Lecture.create("Broken notes", state="paused")
        archived = []

        def archive(job, l):
            archived.append(1)
        with mock.patch.multiple(self.p, **self.stages(lec, archive=archive)), \
                mock.patch.object(self.notes, "to_markdown",
                                  side_effect=RuntimeError("disk full")):
            self.p.submit(lec.id)
            self.wait_done(lec.id)
        self.assertEqual(archived, [1])
        job = self.p.job(lec.id)
        self.assertEqual(job["stage"], "failed")
        self.assertIn("disk full", job["error"])
        self.assertEqual(self.Lecture.load(lec.id).manifest()["state"], "failed")

    def test_a_crash_during_the_archive_is_recovered_as_before(self):
        """Nothing notes-related is persisted before the archive is installed,
        so a crash mid-archive looks exactly as it did with the serial order."""
        self.model()
        lec = self.Lecture.create("Crash", state="paused")
        seen = {}

        def archive(job, l):
            l.set_state("archiving")
            time.sleep(0.2)                   # notes may be done computing by now
            seen["notes_only"] = self.recovery.finish_notes_only(self.Lecture.load(l.id))
            seen["state"] = self.Lecture.load(l.id).manifest()["state"]
            raise SystemExit                  # the process "dies" here
        with mock.patch.multiple(self.p, **self.stages(lec, archive=archive)):
            job = self.pipeline.Job(lec.id)
            self.jobs.append(job)
            t = threading.Thread(target=self.p._run, args=(job,), daemon=True)
            t.start()
            t.join(5)
        self.assertIsNone(seen["notes_only"])     # not mistaken for "only notes left"
        self.assertEqual(seen["state"], "archiving")
        self.assertFalse((lec.path / "notes.json").exists())


    def test_a_failure_after_the_archive_cancels_the_notes_run(self):
        """H2: whatever raises between the archive and saving the notes, the
        background run is cancelled instead of running on, holding the slot."""
        gate = threading.Event()
        fake = self.model(block=gate)
        lec = self.Lecture.create("Fails after archive", state="paused")
        real = self.Lecture.set_state

        def set_state(this, state, **extra):
            if state == "noting":
                raise OSError("manifest write failed")
            return real(this, state, **extra)

        def archive(job, l):
            fake.started.wait(5)
        with mock.patch.multiple(self.p, **self.stages(lec, archive=archive)), \
                mock.patch.object(self.Lecture, "set_state", set_state):
            self.p.submit(lec.id)
            self.wait_done(lec.id)
            run = self.p._jobs[lec.id].notes_run
            self.assertTrue(run.cancel.is_set())
            gate.set()
            self.assertTrue(run.done.wait(5))
        self.assertIn("manifest write failed", self.p.job(lec.id)["error"])
        self.assertLess(len(fake.maps), len(self.notes.chunk(self.doc)))

    def test_a_notes_error_beside_a_failed_archive_is_logged(self):
        """M1: the archive's failure is what the lecture reports; the notes
        thread's own error is not lost."""
        self.model()
        lec = self.Lecture.create("Both fail", state="paused")
        crashed = threading.Event()

        def generate(*a, **k):
            crashed.set()
            raise KeyError("notes broke")

        def archive(job, l):
            crashed.wait(5)
            job.notes_run.done.wait(5)
            raise RuntimeError("archive did not verify")
        with mock.patch.multiple(self.p, **self.stages(lec, archive=archive)), \
                mock.patch.object(self.notes, "generate", generate), \
                self.assertLogs(self.pipeline.log, logging.ERROR) as cm:
            self.p.submit(lec.id)
            self.wait_done(lec.id)
        self.assertIn("archive did not verify", self.p.job(lec.id)["error"])
        self.assertTrue(any("KeyError" in line for line in cm.output), cm.output)

    def test_a_notes_thread_that_cannot_start_falls_back_to_serial(self):
        """M2: never fails the lecture before the archive runs."""
        self.model()
        lec = self.Lecture.create("No thread", state="paused")
        archived = []

        def archive(job, l):
            archived.append(1)
        with mock.patch.multiple(self.p, **self.stages(lec, archive=archive)), \
                mock.patch.object(self.p, "_start_notes",
                                  side_effect=RuntimeError("can't start new thread")):
            self.p.submit(lec.id)
            self.wait_done(lec.id)
        self.assertEqual(archived, [1])
        m = self.Lecture.load(lec.id).manifest()
        self.assertEqual(m["state"], "complete", self.p.job(lec.id))
        self.assertFalse(m["timing"]["overlap"])
        self.assertTrue((lec.path / "notes.json").is_file())

    def test_the_job_lets_go_of_the_cache_and_the_notes(self):
        """L3"""
        self.model()
        lec = self.Lecture.create("Light", state="paused")
        self.storage.atomic_write_json(lec.work / self.live.PREMAP_NAME,
                                       {"schema": 1, "entries": {"k": {"ver": "x"}}})
        with mock.patch.multiple(self.p, **self.stages(lec)):
            self.p.submit(lec.id)
            self.wait_done(lec.id)
        job = self.p._jobs[lec.id]
        self.assertEqual(job.premapped, {})
        self.assertIsNone(job.notes_run.result)


# ============================================================ premap: cache
class Cache(Base):
    def premap_all(self, draft):
        """Map every stable chunk of a draft, as the worker would."""
        entries, skipped = {}, set()
        while True:
            got = self.premap.map_next(draft, entries, skipped)
            if got is None:
                return entries
            entries = {**entries, got[0]: got[1]}

    def assertCitesTrue(self, nd, final):
        """Every cite points at a sentence saying exactly what the note says
        (the fake model's notes are verbatim sentences)."""
        by_sid = {s["sid"]: s for s in final["sents"]}
        n = 0
        for sec in nd["sections"]:
            for b in sec["bullets"]:
                text = b["text"].replace(" " + FakeModel.PARAPHRASE, "")
                for c in b["cites"]:
                    said = self.notes.sentence_text(final, by_sid[c])
                    self.assertEqual(self.norm(said), self.norm(text))
                    n += 1
        return n

    def test_same_text_is_reused_and_cites_follow_the_sentence_ids(self):
        words, t0s = lecture_words(40)
        final = build(words, t0s)
        # The draft's sentence ids are shifted: two extra sentences up front.
        draft = build(["opening.", "remarks."] + words, [0, 100] + t0s)
        self.model()
        entries = self.premap_all(draft)
        self.assertTrue(entries)
        fake = self.model()
        stats = {}
        nd = self.notes.generate(final, premapped=entries, stats=stats)
        n_chunks = len(self.notes.chunk(final))
        hits = stats["premap"]["exact"] + stats["premap"]["fuzzy"]
        self.assertGreaterEqual(hits, n_chunks - 2)
        self.assertEqual(len(fake.maps), n_chunks - hits)
        self.assertGreater(self.assertCitesTrue(nd, final), 0)

    def test_a_stale_cache_is_never_reused(self):
        words, t0s = lecture_words(30)
        doc = build(words, t0s)
        self.model()
        entries = self.premap_all(doc)
        for change in ({"LLM_MODEL": "other:1b"}, {"LLM_NUM_CTX": 4096}):
            with mock.patch.multiple(self.notes.config, **change):
                fake = self.model()
                stats = {}
                self.notes.generate(doc, premapped=entries, stats=stats)
                self.assertEqual(stats["premap"]["exact"] + stats["premap"]["fuzzy"], 0)
                self.assertEqual(len(fake.maps), len(self.notes.chunk(doc)))
        with mock.patch.object(self.premap, "PROMPT_REV", self.premap.PROMPT_REV + 1):
            stats = {}
            self.notes.generate(doc, premapped=entries, stats=stats)
            self.assertEqual(stats["premap"]["exact"] + stats["premap"]["fuzzy"], 0)

    def test_hit_rate_with_segment_seam_differences(self):
        """The honest number: exact hashes almost never match across seams
        (every ~6-minute chunk spans several 60-second seams); the fuzzy
        match (same span, >= FUZZY_MIN_RATIO of the words) does, and every cite
        it keeps still points at a sentence with identical text."""
        words, t0s = lecture_words(50)
        final = build(words, t0s)
        draft = build(*seam_draft(words, t0s))
        self.model()
        entries = self.premap_all(draft)
        fake = self.model()
        stats = {}
        nd = self.notes.generate(final, premapped=entries, stats=stats)
        n = len(self.notes.chunk(final))
        pm = stats["premap"]
        rate = (pm["exact"] + pm["fuzzy"]) / n
        self.assertEqual(pm["exact"], 0)            # said plainly: exact matching alone is useless here
        self.assertGreaterEqual(rate, 0.6, pm)
        self.assertEqual(len(fake.maps), n - pm["exact"] - pm["fuzzy"])
        self.assertCitesTrue(nd, final)

    def test_hit_rate_with_a_paraphrasing_model(self):
        """Words the model adds of its own (in neither transcript) do not
        turn a fuzzy hit into a miss."""
        words, t0s = lecture_words(50)
        final = build(words, t0s)
        draft = build(*seam_draft(words, t0s))
        self.model(paraphrase=True)
        entries = self.premap_all(draft)
        self.model(paraphrase=True)
        stats = {}
        nd = self.notes.generate(final, premapped=entries, stats=stats)
        pm, n = stats["premap"], len(self.notes.chunk(final))
        self.assertGreaterEqual((pm["exact"] + pm["fuzzy"]) / n, 0.6, pm)
        self.assertTrue(any(FakeModel.PARAPHRASE in b["text"]
                            for sec in nd["sections"] for b in sec["bullets"]))
        self.assertCitesTrue(nd, final)

    def test_a_different_passage_is_never_matched(self):
        final = build(*lecture_words(30, seed=0))
        other = build(*lecture_words(30, seed=5))           # same times, other words
        self.model()
        entries = self.premap_all(other)
        stats = {}
        fake = self.model()
        self.notes.generate(final, premapped=entries, stats=stats)
        self.assertEqual(stats["premap"]["exact"] + stats["premap"]["fuzzy"], 0)
        self.assertEqual(len(fake.maps), len(self.notes.chunk(final)))

    def test_a_cite_to_a_changed_sentence_is_dropped_not_moved(self):
        final = build(*lecture_words(20))
        ch = self.notes.chunk(final)[0]
        draft_sents = [[s["sid"] + 5, self.norm(self.notes.sentence_text(final, s))]
                       for s in ch["sents"]]
        draft_sents[3][1] = "a sentence the final pass heard differently"
        entry = {"ver": self.premap.version(), "t0": ch["t0"], "t1": ch["t1"],
                 "sents": draft_sents,
                 "data": {"topic": "", "bullets": [
                     {"text": "x", "cites": [draft_sents[3][0], draft_sents[4][0]]}],
                     "terms": []}}
        data, dropped, whole = self.premap._reground(
            entry, self.premap._sents(final, ch), positional=False)
        self.assertEqual(data["bullets"][0]["cites"], [ch["sents"][4]["sid"]])
        self.assertEqual(dropped, 1)
        self.assertEqual(whole["bullets"], [False])
        # ...and as a fuzzy hit that note is not reused at all (rule: every cite).
        self.assertIsNone(self.premap.lookup({"k": entry}, final, ch, set()))
        self.assertEqual(entry["data"]["bullets"][0]["cites"],
                         [draft_sents[3][0], draft_sents[4][0]])   # cache untouched

    def fuzzy_entry(self, final, ch, bullets, *, shift=5, sents=None):
        sents = sents or [[s["sid"] + shift, self.norm(self.notes.sentence_text(final, s))]
                          for s in ch["sents"]]
        sents[1][1] += " extra"                    # not an exact match
        return {"ver": self.premap.version(), "t0": ch["t0"], "t1": ch["t1"],
                "sents": sents, "data": {"topic": "", "bullets": bullets, "terms": []}}

    def test_draft_only_wording_is_dropped_on_a_fuzzy_hit(self):
        """H1: the draft heard "fifteen" where the final pass heard "fifty".
        A note using the draft's word passes lecture-wide grounding ("fifteen"
        is said elsewhere) but is dropped; a note in the model's own words
        (in neither transcript) is kept, as for a freshly mapped part."""
        words, t0s = lecture_words(20)
        words[40:42] = ["fifty", "kilojoules."]
        words += ["fifteen."]                        # said elsewhere in the lecture
        t0s = t0s + [t0s[-1] + 400]
        final = build(words, t0s)
        ch = self.notes.chunk(final)[0]
        sents = [[s["sid"] + 5, self.norm(self.notes.sentence_text(final, s))]
                 for s in ch["sents"]]
        heard = next(i for i, x in enumerate(sents) if "fifty" in x[1])
        sents[heard][1] = sents[heard][1].replace("fifty", "fifteen")
        cite = sents[heard][0]
        same = sents[0][0] if heard else sents[1][0]          # a sentence both heard alike
        good = self.notes.sentence_text(final, ch["sents"][heard])
        bullets = [{"text": "Energy released totals fifty kilojoules overall", "cites": [cite]},
                   {"text": f"{good} elucidated thoroughly", "cites": [same]},
                   {"text": "Energy released totals fifteen kilojoules", "cites": [same]},
                   {"text": "Energy released totals 15 kilojoules", "cites": [same]}]
        for k in (heard + 1, heard + 2):                           # enough survives for a hit
            bullets.append({"text": self.notes.sentence_text(final, ch["sents"][k]),
                            "cites": [sents[k][0]]})
        entry = {"ver": self.premap.version(), "t0": ch["t0"], "t1": ch["t1"],
                 "sents": sents, "data": {"topic": "", "bullets": bullets, "terms": []}}
        data, kind, _ = self.premap.lookup({"k": entry}, final, ch, set())
        texts = [b["text"] for b in data["bullets"]]
        self.assertEqual(kind, "fuzzy")
        self.assertIn(f"{good} elucidated thoroughly", texts)      # paraphrase survives
        self.assertNotIn("Energy released totals fifteen kilojoules", texts)
        self.assertNotIn("Energy released totals 15 kilojoules", texts)
        # The cite to the misheard sentence is dropped (its text differs), so
        # even the correct "fifty" note goes: no cite left in this part.
        self.assertNotIn("Energy released totals fifty kilojoules overall", texts)

    def vet_case(self, final_sentence, draft_sentence, texts, cite=0):
        """A fuzzy hit on part 0 where the draft heard one sentence (the 4th)
        differently. Each of `texts` cites sentence `cite`; four verbatim
        notes keep enough of the part alive for a hit. Returns the notes kept."""
        words, t0s = lecture_words(20)
        cut = [i for i, w in enumerate(words) if w.endswith(".")][2] + 1
        ins = final_sentence.split()
        words = words[:cut] + ins + words[cut:]
        t0s = t0s[:cut] + [t0s[cut - 1] + 40 * (k + 1) for k in range(len(ins))] + t0s[cut:]
        final = build(words, t0s)
        ch = self.notes.chunk(final)[0]
        self.assertEqual(self.notes.sentence_text(final, ch["sents"][3]), final_sentence)
        sents = [[x["sid"] + 5, self.norm(self.notes.sentence_text(final, x))]
                 for x in ch["sents"]]
        sents[3][1] = self.norm(draft_sentence)
        cites = cite if isinstance(cite, list) else [cite]
        bullets = [{"text": t, "cites": [sents[c][0] for c in cites]} for t in texts]
        bullets += [{"text": self.notes.sentence_text(final, ch["sents"][k]),
                     "cites": [sents[k][0]]} for k in (4, 5, 6, 7)]
        entry = {"ver": self.premap.version(), "t0": ch["t0"], "t1": ch["t1"],
                 "sents": sents, "data": {"topic": "", "bullets": bullets, "terms": []}}
        hit = self.premap.lookup({"k": entry}, final, ch, set())
        self.assertIsNotNone(hit)
        self.assertEqual(hit[1], "fuzzy")
        return [b["text"] for b in hit[0]["bullets"]]

    def test_reviewer_repro_numbers_and_short_acronyms(self):
        kept = self.vet_case("glycolysis yields 2 atp per glucose.",
                             "glycolysis yields 4 adp per glucose.",
                             ["Glycolysis yields 4 ADP per glucose",
                              "Glycolysis yields 2 ATP per glucose"])
        self.assertNotIn("Glycolysis yields 4 ADP per glucose", kept)
        self.assertIn("Glycolysis yields 2 ATP per glucose", kept)

    def test_reviewer_repro_negation(self):
        kept = self.vet_case("the enzyme does bind oxygen.",
                             "the enzyme does not bind oxygen.",
                             ["The enzyme does not bind oxygen",
                              "The enzyme binds oxygen readily"])
        self.assertNotIn("The enzyme does not bind oxygen", kept)
        self.assertIn("The enzyme binds oxygen readily", kept)    # paraphrase is fine

    def test_a_single_digit_only_the_draft_heard(self):
        kept = self.vet_case("there are 3 stages.", "there are 5 stages.",
                             ["There are 5 stages", "There are 3 stages"])
        self.assertEqual([t for t in kept if "stages" in t], ["There are 3 stages"])

    def test_a_decimal_only_the_draft_heard(self):
        kept = self.vet_case("the ph is 7.4 here.", "the ph is 7.2 here.",
                             ["The pH is 7.2", "The pH is 7.4"])
        self.assertEqual([t for t in kept if "pH" in t], ["The pH is 7.4"])

    def test_an_acronym_only_the_draft_heard(self):
        kept = self.vet_case("dna is copied.", "rna is copied.",
                             ["RNA is copied", "DNA is copied"])
        self.assertEqual([t for t in kept if "copied" in t], ["DNA is copied"])

    def test_a_negation_word_absent_from_the_final_part(self):
        kept = self.vet_case("it always reacts.", "it never reacts.",
                             ["It never reacts", "It always reacts"])
        self.assertEqual([t for t in kept if "reacts" in t], ["It always reacts"])

    def test_a_note_leaning_on_a_changed_sentence_is_dropped(self):
        """Every original cite must survive: one dropped cite means the note
        leans on a sentence whose text changed."""
        kept = self.vet_case("dna is copied.", "rna is copied.",
                             ["Copying happens"], cite=[0, 3])
        self.assertNotIn("Copying happens", kept)

    def test_a_bullet_without_a_surviving_cite_is_dropped(self):
        final = build(*lecture_words(20))
        ch = self.notes.chunk(final)[0]
        s3, s4 = ch["sents"][3], ch["sents"][4]
        entry = self.fuzzy_entry(final, ch, [
            {"text": self.notes.sentence_text(final, s3), "cites": [s3["sid"] + 5]},
            {"text": self.notes.sentence_text(final, s4), "cites": [999]}])
        data, _, _ = self.premap.lookup({"k": entry}, final, ch, set())
        self.assertEqual([b["cites"] for b in data["bullets"]], [[s3["sid"]]])

    def test_a_longer_draft_part_does_not_bring_bullets_from_the_next_part(self):
        """The draft part runs on into the next final part: its notes about
        that stretch must not be reused for this part."""
        final = build(*lecture_words(30))
        chunks = self.notes.chunk(final)
        ch, nxt = chunks[0], chunks[1]
        tail = [s for s in nxt["sents"] if s["sid"] > ch["sents"][-1]["sid"]][:4]
        sents = [[s["sid"] + 5, self.norm(self.notes.sentence_text(final, s))]
                 for s in ch["sents"] + tail]
        inside, beyond = ch["sents"][3], tail[2]
        entry = self.fuzzy_entry(final, ch, [
            {"text": self.notes.sentence_text(final, inside), "cites": [inside["sid"] + 5]},
            {"text": self.notes.sentence_text(final, beyond), "cites": [beyond["sid"] + 5]}],
            sents=sents)
        entry["t1"] = tail[-1]["t1"]
        data, kind, _ = self.premap.lookup({"k": entry}, final, ch, set())
        self.assertEqual(kind, "fuzzy")
        self.assertEqual([b["cites"] for b in data["bullets"]], [[inside["sid"]]])

    def test_too_little_surviving_is_a_miss(self):
        final = build(*lecture_words(20))
        ch = self.notes.chunk(final)[0]
        s3 = ch["sents"][3]
        good = self.notes.sentence_text(final, s3)
        entry = self.fuzzy_entry(final, ch, [
            {"text": good, "cites": [s3["sid"] + 5]},
            {"text": f"fifteen {good}", "cites": [s3["sid"] + 5]},
            {"text": f"seventeen {good}", "cites": [s3["sid"] + 5]}])
        used = set()
        self.assertIsNone(self.premap.lookup({"k": entry}, final, ch, used))
        self.assertEqual(used, set())

    def test_end_to_end_draft_wording_never_reaches_the_notes(self):
        words, t0s = lecture_words(50)
        for i in range(3, len(words), 25):          # the final pass heard "fifty"...
            words[i] = "fifty." if words[i].endswith(".") else "fifty"
        final = build(words + ["fifteen."], t0s + [t0s[-1] + 400])
        dw, dt = seam_draft(words, t0s)             # ...the draft heard "fifteen"
        draft = build([w.replace("fifty", "fifteen") for w in dw], dt)
        self.model(inject="fifteen", paraphrase=True)
        entries = self.premap_all(draft)
        self.model()
        stats = {}
        nd = self.notes.generate(final, premapped=entries, stats=stats)
        self.assertGreater(stats["premap"]["fuzzy"], 0)
        for sec in nd["sections"]:
            for b in sec["bullets"]:
                self.assertNotIn("fifteen", b["text"])
        self.assertCitesTrue(nd, final)

    def test_the_time_budget_counts_only_parts_the_model_will_run(self):
        """M3: cached parts cost nothing, so they do not count toward the
        projection that cuts a slow run short."""
        doc = build(*lecture_words(50))
        self.model()
        entries = self.premap_all(doc)
        chunks = self.notes.chunk(doc)
        ver = self.premap.version()
        for ch in chunks[:2]:                      # these two need the model again
            entries.pop(self.premap._key(ver, self.premap._sents(doc, ch)))
        fake = self.model(delay=0.3)
        with mock.patch.object(self.notes, "NOTES_BUDGET_S", 2):
            nd = self.notes.generate(doc, premapped=entries)
        self.assertIsNone(nd.get("partial_from_ms"))
        self.assertEqual(len(fake.maps), 3)        # two uncached and the last part

    def test_only_stable_chunks_are_premapped(self):
        draft = build(*lecture_words(20))
        self.model()
        entries = self.premap_all(draft)
        self.assertEqual(len(entries), len(self.notes.chunk(draft)) - 1)


# =========================================================== premap: worker
class Worker(Base):
    def setUp(self):
        super().setUp()
        self.lec = self.Lecture.create(title="rec")
        self.draft = build(*lecture_words(20))
        self.storage.atomic_write_json(self.lec.draft_path, self.draft)
        self.state = "go"
        p = mock.patch.object(self.live.LIVE, "premap_state",
                              lambda lid: self.state)
        p.start()
        self.addCleanup(p.stop)
        self.P = self.premap.PreMapper()
        self.addCleanup(self.stop_all)
        poll = mock.patch.object(self.premap, "POLL_S", 0.02)
        poll.start()
        self.addCleanup(poll.stop)

    def stop_all(self):
        for lid in list(self.P._sessions):
            self.P.stop(lid)
        t = self.P._thread
        if t is not None:
            t.join(5)

    def cache(self):
        return self.premap.load(self.lec)

    def stable(self):
        return len(self.notes.chunk(self.draft)) - 1

    def test_maps_stable_parts_into_the_work_dir(self):
        fake = self.model()
        self.P.begin(self.lec.id)
        wait_for(lambda: len(self.cache()) == self.stable(), what="cache filled")
        self.assertEqual(len(fake.maps), self.stable())
        self.assertTrue((self.lec.work / self.live.PREMAP_NAME).is_file())

    def test_switched_off_by_env(self):
        fake = self.model()
        with mock.patch.dict(os.environ, {"LECTERN_PREMAP": "0"}):
            self.P.begin(self.lec.id)
            wait_for(lambda: not self.P._sessions, what="session dropped")
        self.assertEqual(fake.maps, [])

    def test_not_on_a_small_machine_or_without_the_model(self):
        fake = self.model()
        self.RAM = 8 * GB
        self.P.begin(self.lec.id)
        wait_for(lambda: not self.P._sessions, what="session dropped (RAM)")
        self.RAM = 32 * GB
        with mock.patch.object(self.notes, "available", lambda: False):
            self.P.begin(self.lec.id)
            wait_for(lambda: not self.P._sessions, what="session dropped (no model)")
        self.assertEqual(fake.maps, [])
        self.assertEqual(self.cache(), {})

    def test_yields_while_live_is_behind_and_ends_when_it_is_off(self):
        fake = self.model()
        self.state = "wait"
        self.P.begin(self.lec.id)
        time.sleep(0.2)
        self.assertEqual(fake.maps, [])
        self.assertTrue(self.P._sessions)
        self.state = "end"                       # live switched itself off
        wait_for(lambda: not self.P._sessions, what="session ended")
        self.assertEqual(fake.maps, [])

    def test_never_beside_an_audio_stage_or_other_notes(self):
        fake = self.model()
        for lock in (self.pipeline.STAGE_LOCK, self.pipeline.NOTES_LOCK):
            with lock:
                self.P.begin(self.lec.id)
                time.sleep(0.2)
                self.assertEqual(fake.maps, [], lock)
            self.stop_all()

    def test_stop_is_prompt_and_nothing_is_written_after_it(self):
        gate = threading.Event()
        fake = self.model(block=gate)
        self.P.begin(self.lec.id)
        self.assertTrue(fake.started.wait(5))
        self.P.stop(self.lec.id)
        self.assertFalse(fake.returned.is_set())   # stop did not wait for the call
        gate.set()
        t = self.P._thread
        if t is not None:
            t.join(5)
        self.assertEqual(self.cache(), {})

    def unload_spy(self, block=None):
        calls = threading.Event()

        def unload():
            calls.set()
            if block is not None:
                block.wait(10)
        p = mock.patch.object(self.notes, "_unload", unload)
        p.start()
        self.addCleanup(p.stop)
        return calls

    def test_stop_does_not_unload_the_model(self):
        """M4: the notes after Stop need the model within seconds."""
        unloaded = self.unload_spy()
        fake = self.model()
        self.P.begin(self.lec.id)
        wait_for(lambda: fake.maps, what="a pre-mapped part")
        self.P.stop(self.lec.id)
        t = self.P._thread
        if t is not None:
            t.join(5)
        self.assertFalse(unloaded.is_set())

    def test_the_model_is_unloaded_when_premapping_ends_mid_recording(self):
        """M4: live switched itself off (or a pause that is not resumed):
        the model is freed, from a thread, so the worker never waits."""
        gate = threading.Event()
        self.addCleanup(gate.set)
        unloaded = self.unload_spy(block=gate)
        fake = self.model()
        self.P.begin(self.lec.id)
        wait_for(lambda: fake.maps, what="a pre-mapped part")
        self.state = "end"
        self.assertTrue(unloaded.wait(5))
        wait_for(lambda: self.P._thread is None, what="worker exit while unload blocks")

    def test_unload_waits_for_the_last_session_even_if_it_made_no_call(self):
        """N3: a call made for one recording is unloaded when the LAST
        pre-mapping session is dropped, whichever made the call."""
        unloaded = self.unload_spy()
        fake = self.model()
        other = self.Lecture.create(title="other")
        self.P.begin(self.lec.id)
        wait_for(lambda: fake.maps, what="a pre-mapped part")
        states = {self.lec.id: "end", other.id: "wait"}
        with mock.patch.object(self.live.LIVE, "premap_state", lambda lid: states[lid]):
            self.P.begin(other.id)
            wait_for(lambda: self.lec.id not in self.P._sessions, what="first dropped")
            self.assertFalse(unloaded.is_set())     # the other recording is still on
            states[other.id] = "end"
            wait_for(lambda: not self.P._sessions, what="second dropped")
        self.assertTrue(unloaded.wait(5))

    def test_no_unload_when_no_call_was_made(self):
        unloaded = self.unload_spy()
        self.model()
        self.state = "wait"
        self.P.begin(self.lec.id)
        time.sleep(0.1)
        self.state = "end"
        wait_for(lambda: not self.P._sessions, what="dropped")
        time.sleep(0.1)
        self.assertFalse(unloaded.is_set())

    def test_the_stop_window_does_not_unload(self):
        """N1: through the real live state. Stop marks capture ended, then the
        pipeline's finish() removes the live session, and only then calls
        PREMAP.stop(); the worker reading the state in between must not
        unload the model the notes are about to use."""
        unloaded = self.unload_spy()
        fake = self.model()
        real = self.live.LiveTranscriber.premap_state.__get__(self.live.LIVE)
        sess = self.live._Session(self.lec.id)
        self.live.LIVE._sessions[self.lec.id] = sess
        self.addCleanup(self.live.LIVE._sessions.pop, self.lec.id, None)
        (self.lec.work / "seg-000000.wav").write_bytes(b"")
        with mock.patch.object(self.live.LIVE, "premap_state", real):
            self.P.begin(self.lec.id)
            wait_for(lambda: fake.maps, what="a pre-mapped part")
            sess.capturing = False                       # capture_ended (Stop)
            time.sleep(0.1)
            self.assertIn(self.lec.id, self.P._sessions)  # waiting, not dropped
            del self.live.LIVE._sessions[self.lec.id]    # finish()
            wait_for(lambda: not self.P._sessions, what="dropped at stop")
            t = self.P._thread
            if t is not None:
                t.join(5)
        self.assertFalse(unloaded.is_set())

    def test_a_pause_through_the_real_live_state_unloads(self):
        unloaded = self.unload_spy()
        fake = self.model()
        real = self.live.LiveTranscriber.premap_state.__get__(self.live.LIVE)
        sess = self.live._Session(self.lec.id)
        self.live.LIVE._sessions[self.lec.id] = sess
        self.addCleanup(self.live.LIVE._sessions.pop, self.lec.id, None)
        (self.lec.work / "seg-000000.wav").write_bytes(b"")
        with mock.patch.object(self.live.LIVE, "premap_state", real):
            self.P.begin(self.lec.id)
            wait_for(lambda: fake.maps, what="a pre-mapped part")
            sess.capturing = False
            self.lec.set_state("paused")
            self.assertTrue(unloaded.wait(5))

    def test_premap_call_timeout_is_proportionate(self):
        self.assertLessEqual(self.premap.CALL_TIMEOUT_S, 180)

    def test_a_quick_pause_and_resume_keeps_premapping_on(self):
        """L1: the worker saw "paused" and is about to drop the session just
        as the recording resumes; the resumed recording keeps pre-mapping."""
        fake = self.model()
        seen = []

        def state(lid):
            seen.append(1)
            if len(seen) == 1:
                self.P.begin(lid)                 # resume, racing the drop
                return "end"
            return "go"
        with mock.patch.object(self.live.LIVE, "premap_state", state):
            self.P.begin(self.lec.id)
            wait_for(lambda: fake.maps, what="pre-mapping after resume")

    def test_a_model_failure_is_silent_and_leaves_the_normal_path(self):
        fake = self.model(fail=True)
        self.P.begin(self.lec.id)
        wait_for(lambda: not self.P._sessions, what="session dropped after failure")
        self.assertEqual(self.cache(), {})
        ok = self.model()
        nd = self.notes.generate(self.draft, premapped=self.cache())
        self.assertEqual(nd["engine"], self.config.LLM_MODEL)
        self.assertEqual(len(ok.maps), len(self.notes.chunk(self.draft)))

    def test_cache_goes_with_the_draft_and_the_trash_purge_allows_it(self):
        self.storage.atomic_write_json(self.lec.work / self.live.PREMAP_NAME, {"x": 1})
        self.live.discard(self.lec)
        self.assertFalse((self.lec.work / self.live.PREMAP_NAME).exists())
        self.assertIn(self.live.PREMAP_NAME, self.recovery._DELETABLE_NAMES)

    def test_a_corrupt_cache_reads_as_empty(self):
        (self.lec.work / self.live.PREMAP_NAME).write_text("{not json")
        self.assertEqual(self.cache(), {})


class LiveState(Base):
    """live.LIVE.premap_state: the yield rule, from the live session itself."""

    def test_states(self):
        LIVE = self.live.LiveTranscriber()
        lec = self.Lecture.create(title="s")
        self.assertEqual(LIVE.premap_state(lec.id), "stop")         # no live session
        s = self.live._Session(lec.id)
        LIVE._sessions[lec.id] = s
        for n in range(3):
            (lec.work / f"seg-{n:06d}.wav").write_bytes(b"")
        s.segments = [{"file": "seg-000000.wav"}]
        self.assertEqual(LIVE.premap_state(lec.id), "wait")        # one finished segment pending
        s.segments.append({"file": "seg-000001.wav"})
        self.assertEqual(LIVE.premap_state(lec.id), "go")          # caught up; 2 is being written
        LIVE._current = s
        self.assertEqual(LIVE.premap_state(lec.id), "wait")        # a live engine is running
        LIVE._current = None
        s.off = "too slow to keep up"
        self.assertEqual(LIVE.premap_state(lec.id), "end")         # live switched itself off
        s.off, s.capturing = None, False
        self.assertEqual(LIVE.premap_state(lec.id), "wait")        # Stop or Pause: not known yet
        lec.set_state("paused")
        self.assertEqual(LIVE.premap_state(lec.id), "end")         # a pause
        lec.set_state("recording")
        s.finishing = True
        self.assertEqual(LIVE.premap_state(lec.id), "stop")        # the pipeline is finishing it
        del LIVE._sessions[lec.id]
        self.assertEqual(LIVE.premap_state(lec.id), "stop")        # finished (or never began)


class PipelineUsesCache(PipelineHarness):
    def test_cache_is_loaded_after_stop_and_reused(self):
        lec = self.Lecture.create("Reuse", state="paused")
        self.model()
        entries = Cache.premap_all(self, self.doc)
        self.storage.atomic_write_json(lec.work / self.live.PREMAP_NAME,
                                       {"schema": 1, "entries": entries})
        fake = self.model()
        stops = []
        real_transcribe = self.stages(lec)["_transcribe"]

        def transcribe(job, l):
            real_transcribe(job, l)
            self.live.discard(l)                    # as the real stage does
        st = {**self.stages(lec), "_transcribe": transcribe}
        with mock.patch.multiple(self.p, **st), \
                mock.patch.object(self.premap.PREMAP, "stop", stops.append):
            self.p.submit(lec.id)
            self.wait_done(lec.id)
        self.assertEqual(stops, [lec.id])
        m = self.Lecture.load(lec.id).manifest()
        self.assertEqual(m["state"], "complete")
        self.assertEqual(m["notes"]["premap"]["exact"], len(entries))
        self.assertEqual(len(fake.maps), len(self.notes.chunk(self.doc)) - len(entries))
        self.assertFalse((lec.work / self.live.PREMAP_NAME).exists())


if __name__ == "__main__":
    unittest.main()
