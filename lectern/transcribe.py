"""whisper.cpp invocation and normalization to the click-to-jump contract.

Flags, and why each one is absent:
  * no -dtw   : whisper.cpp 1.9.1 defaults flash_attn=true, which silently
                disables DTW. Verified: every t_dtw comes back -1.
  * no -ml 1 -sow : -ojf alone already emits per-token ms offsets inside
                readable sentence segments, so word-splitting is pure loss.
  * no --vad  : output_json reads raw token structs that whisper never remaps
                back to the original timeline.

Flags, and why each one is present:
  * -mc 0     : no previous-window text as context. With context, one bad
                window poisons every later one: measured on a damaged lecture,
                the default emitted ONE sentence 291 times and nothing else;
                -mc 0 -sns recovered 48 distinct real sentences.
  * -sns      : suppress non-speech tokens, whisper's own "*music*" /
                "*audio: ...*" narration. Same default as OpenAI's whisper.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
from pathlib import Path

from . import config, hardware, models
from . import platform_compat as pc

# Whisper special tokens: [_BEG_], [_EOT_], [_SOT_], and timestamp tokens
# like [_TT_519] which end in a digit, not an underscore.
_SPECIAL = re.compile(r"^\[_[A-Za-z]+_?\d*_?\]$")
_SENT_END = re.compile(r'[.!?]["\')\]]?$')
# Whisper narrates non-speech as ordinary text on silent or musical stretches
# ("*music*", "(coughing)", "[BLANK_AUDIO]"). Measured on a real recording with
# background music: it produced ten "*music*" words and nothing else useful.
# These are annotations, never lecture content, so they are dropped.
# Deliberately narrow: "part (a)" and a cited "[12]" are real lecture content
# and must survive. Only whisper's own annotation conventions are dropped.
_ANNOTATION = re.compile(
    r'^(?:'
    r'\*.+?\*'                    # *music*, *applause*
    r'|\[[A-Z][A-Z_ ]{2,}\]'      # [BLANK_AUDIO], [MUSIC PLAYING]
    r'|\([a-z][a-z ]{3,}\)'       # (coughing), (laughter) - never "(a)", "(12)"
    r')[.,!?]?$')
MAX_WORDS_PER_SENT = 45          # guards a stretch with no punctuation
MAX_ANNOTATION_WORDS = 12        # longest "*audio: "I'm sorry"*" span we drop


class TranscriptionError(RuntimeError):
    pass


def run_whisper(wav: Path, out_prefix: Path, *, progress_cb=None) -> Path:
    model = models.active_path()
    if model is None:
        raise TranscriptionError(
            "no speech model is installed. Open Settings and download one "
            f"(or put a ggml-*.bin file in {models.MODELS_DIR}).")
    if not model.is_file():
        raise TranscriptionError(f"model not found: {model}")
    argv = [
        config.WHISPER_CLI,
        "-m", str(model),
        "-f", str(wav),
        "-ojf",
        "-of", str(out_prefix),
        "-t", str(hardware.whisper_threads()),
        "-bs", str(config.WHISPER_BEAM),
        "-l", config.WHISPER_LANG,
        "-mc", "0",
        "-sns",
        "-pp",
    ]

    # Below-normal priority: on a weak machine a full-tilt whisper starves the
    # browser and, worse, the recorder of the NEXT lecture, which drops audio.
    # Priority is inherited, so it is set on the child, never on the backend.
    argv, extra = pc.low_priority(argv)
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True, errors="replace",
                                bufsize=1, **extra)
    except FileNotFoundError:
        raise TranscriptionError(
            "whisper-cli is not installed (on a Mac: brew install whisper-cpp)") from None
    # A hung whisper (GPU stall) holds the stage lock and would block every
    # later lecture forever, so it gets a generous deadline and is killed.
    deadline_s = _whisper_deadline_s(wav)
    timer = threading.Timer(deadline_s, proc.kill)
    timer.daemon = True
    timer.start()
    tail: list[str] = []
    try:
        for line in proc.stderr:                   # type: ignore[union-attr]
            tail.append(line)
            del tail[:-40]
            if progress_cb:
                m = re.search(r"progress\s*=\s*(\d+)%", line)
                if m:
                    try:
                        progress_cb(int(m.group(1)))
                    except Exception:
                        pass
        rc = proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:                    # never leave an orphan
            proc.kill()
            proc.wait()
    out = Path(str(out_prefix) + ".json")
    if rc != 0 or not out.is_file():
        why = (f"whisper-cli took longer than {deadline_s / 60:.0f} min and was stopped"
               if rc in (-9, 137) else f"whisper-cli exited {rc}")
        raise TranscriptionError(why + "\n" + "".join(tail)[-1500:])
    return out


_WAV_BYTES_PER_S = 32_000            # 16 kHz mono s16le
_DEADLINE_FLOOR_S = 20 * 60
_DEADLINE_X_REALTIME = 4             # slowest CPU model measured ~1.2x real time


def _whisper_deadline_s(wav: Path) -> float:
    try:
        dur = wav.stat().st_size / _WAV_BYTES_PER_S
    except OSError:
        dur = 0.0
    return _DEADLINE_FLOOR_S + dur * _DEADLINE_X_REALTIME


def normalize(raw_json: Path, duration_ms: int) -> dict:
    """Turn whisper's -ojf output into the compact transcript contract.

    whisper tokenizes with a leading-space convention: a token whose text starts
    with a space begins a new word; everything else (punctuation, sub-word
    pieces) attaches to the current word.
    """
    try:
        data = json.loads(raw_json.read_text())
    except json.JSONDecodeError as e:
        # whisper writes its JSON with a plain streaming ofstream, so a crash
        # mid-write leaves an unparseable file. Never treat it as a deliverable.
        raise TranscriptionError(f"whisper JSON is unparseable: {e}") from e

    w_text: list[str] = []
    w_t0: list[int] = []
    w_t1: list[int] = []
    w_p: list[float] = []

    for seg in data.get("transcription", []):
        for tok in seg.get("tokens", []):
            text = tok.get("text", "")
            if not text or _SPECIAL.match(text.strip()):
                continue
            off = tok.get("offsets") or {}
            t0 = int(off.get("from", 0))
            t1 = int(off.get("to", t0))
            p = float(tok.get("p", 1.0))
            starts_word = text.startswith(" ") or not w_text
            piece = text.strip()
            if not piece:
                continue
            if starts_word:
                w_text.append(piece)
                w_t0.append(t0)
                w_t1.append(t1)
                w_p.append(p)
            else:
                w_text[-1] += piece
                w_t1[-1] = max(w_t1[-1], t1)
                w_p[-1] = min(w_p[-1], p)

    # Annotations must be filtered AFTER assembly: whisper splits "*music*" into
    # the separate tokens "*", "music", "*", so the complete word only exists here.
    if w_text:
        drop = {i for i, w in enumerate(w_text) if _ANNOTATION.match(w)}
        drop |= _annotation_spans(w_text)
        keep = [i for i in range(len(w_text)) if i not in drop]
        if len(keep) != len(w_text):
            w_text = [w_text[i] for i in keep]
            w_t0 = [w_t0[i] for i in keep]
            w_t1 = [w_t1[i] for i in keep]
            w_p = [w_p[i] for i in keep]

    # Monotonicity: a seek target must never go backwards.
    for i in range(1, len(w_t0)):
        if w_t0[i] < w_t0[i - 1]:
            w_t0[i] = w_t0[i - 1]
        if w_t1[i] < w_t0[i]:
            w_t1[i] = w_t0[i]

    w_text, w_t0, w_t1, w_p = _collapse_repeats(w_text, w_t0, w_t1, w_p)

    sents = []
    start = 0
    for i, word in enumerate(w_text):
        if _SENT_END.search(word) or (i - start + 1) >= MAX_WORDS_PER_SENT:
            sents.append({"sid": len(sents) + 1, "w0": start, "wn": i - start + 1,
                          "t0": w_t0[start], "t1": w_t1[i]})
            start = i + 1
    if start < len(w_text):
        sents.append({"sid": len(sents) + 1, "w0": start,
                      "wn": len(w_text) - start,
                      "t0": w_t0[start], "t1": w_t1[-1]})

    return {
        "schema": 1,
        "duration_ms": int(duration_ms or (w_t1[-1] if w_t1 else 0)),
        "language": config.WHISPER_LANG,
        "words": {"w": w_text, "t0": w_t0, "t1": w_t1,
                  "p": [round(x, 2) for x in w_p]},
        "sents": sents,
    }


def _annotation_spans(words: list[str]) -> set[int]:
    """Multi-word annotations: '*audio:' '"I'm' 'sorry"*' is one narration that
    the single-word pattern above cannot see. Measured on a real recording: the
    whole transcript was this span repeated ~50 times. A span must open with a
    word starting '*' and close within MAX_ANNOTATION_WORDS with a word ending
    '*' (optionally followed by punctuation); an emphasised '*word*' is already
    handled above, and a lone '*' never opens a span on its own."""
    out: set[int] = set()
    i = 0
    while i < len(words):
        w = words[i]
        if w.startswith("*") and len(w) > 1 and not _ANNOTATION.match(w):
            for j in range(i + 1, min(len(words), i + MAX_ANNOTATION_WORDS)):
                if words[j].rstrip('.,!?"\'').endswith("*"):
                    out.update(range(i, j + 1))
                    i = j
                    break
        i += 1
    return out


MIN_LOOP_RUN = 3            # 3+ identical consecutive sentences is pathological
MIN_LOOP_WORDS = 4          # ignore short interjections a lecturer may repeat


def _collapse_repeats(w_text, w_t0, w_t1, w_p):
    """Remove whisper's repetition loops.

    On trailing silence or room noise, whisper.cpp can lock into emitting the
    same sentence over and over -- measured here as one sentence repeated 20+
    times across the last 25 seconds of a recording, which then poisons the
    notes as well. A lecturer does not say the same 4+ word sentence three times
    in a row, so a run of three or more identical sentences is a loop: keep the
    first and drop the rest.
    """
    if not w_text:
        return w_text, w_t0, w_t1, w_p

    # provisional sentence split, purely to find the runs
    spans, start = [], 0
    for i, word in enumerate(w_text):
        if _SENT_END.search(word) or (i - start + 1) >= MAX_WORDS_PER_SENT:
            spans.append((start, i))
            start = i + 1
    if start < len(w_text):
        spans.append((start, len(w_text) - 1))

    key = [" ".join(w_text[a:b + 1]).strip().lower() for a, b in spans]
    drop = set()
    i = 0
    while i < len(spans):
        j = i
        while j + 1 < len(spans) and key[j + 1] == key[i]:
            j += 1
        run = j - i + 1
        if run >= MIN_LOOP_RUN and len(key[i].split()) >= MIN_LOOP_WORDS:
            for k in range(i + 1, j + 1):          # keep the first occurrence
                drop.update(range(spans[k][0], spans[k][1] + 1))
        i = j + 1

    if not drop:
        return w_text, w_t0, w_t1, w_p
    keep = [i for i in range(len(w_text)) if i not in drop]
    return ([w_text[i] for i in keep], [w_t0[i] for i in keep],
            [w_t1[i] for i in keep], [w_p[i] for i in keep])


def sentence_text(t: dict, s: dict) -> str:
    w = t["words"]["w"]
    return " ".join(w[s["w0"]: s["w0"] + s["wn"]])


def to_plain_text(t: dict, per_para: int = 6) -> str:
    """Paragraphed reading copy with a timestamp per paragraph."""
    out, buf = [], []
    for i, s in enumerate(t["sents"]):
        buf.append(sentence_text(t, s))
        if len(buf) >= per_para:
            ms = t["sents"][i - len(buf) + 1]["t0"]
            out.append(f"[{ms//60000:02d}:{ms//1000%60:02d}] " + " ".join(buf))
            buf = []
    if buf:
        ms = t["sents"][-len(buf)]["t0"]
        out.append(f"[{ms//60000:02d}:{ms//1000%60:02d}] " + " ".join(buf))
    return "\n\n".join(out) + "\n"
