"""One-file Markdown export: notes + transcript, ready to paste into Obsidian,
Notion, Apple Notes, NotebookLM, ChatGPT -- anything that reads Markdown.

Built on demand from the files already on disk; nothing new is stored."""
from __future__ import annotations

import json
from datetime import datetime

from .storage import Lecture

PARA_SECONDS = 45          # transcript paragraph length when no pauses help


def _clock(ms) -> str:
    s = max(0, int((ms or 0) // 1000))
    h, m, x = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{x:02d}" if h else f"{m:02d}:{x:02d}"


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _one_line(s) -> str:
    return " ".join(str(s or "").split())


def build_markdown(lec: Lecture) -> str | None:
    """Return the Markdown document, or None if there is nothing to export."""
    notes = _read_json(lec.path / "notes.json")
    tr = _read_json(lec.transcript_path)
    if not notes and not tr:
        return None

    m = lec.manifest()
    title = m.get("title") or lec.id
    out: list[str] = [f"# {title}", ""]

    meta = []
    if m.get("course"):
        meta.append(m["course"])
    created = m.get("created_utc")
    if created:
        try:
            meta.append(datetime.fromisoformat(created.replace("Z", "+00:00"))
                        .astimezone().strftime("%B %d, %Y"))
        except ValueError:
            pass
    if lec.duration_ms():
        meta.append(_clock(lec.duration_ms()) + " long")
    if meta:
        out += ["*" + " · ".join(meta) + "*", ""]

    if notes:
        if notes.get("summary"):
            out += ["## Summary", "", _one_line(notes["summary"]), ""]
        if notes.get("fallback") and notes.get("sections"):
            out += ["> These notes were pulled straight from the transcript "
                    "(no language model was available).", ""]
        for sec in notes.get("sections") or []:
            out += [f"## {_one_line(sec.get('heading'))}", ""]
            for b in sec.get("bullets") or []:
                flag = " (?)" if b.get("grounded") is not None and b["grounded"] < 0.5 else ""
                out.append(f"- `{_clock(b.get('t_ms'))}` {_one_line(b.get('text'))}{flag}")
            out.append("")
        if notes.get("terms"):
            out += ["## Key terms", ""]
            for t in notes["terms"]:
                out.append(f"- **{_one_line(t.get('term'))}** — "
                           f"{_one_line(t.get('definition'))} (`{_clock(t.get('t_ms'))}`)")
            out.append("")
        if notes.get("quiz"):
            out += ["## Check yourself", ""]
            for i, q in enumerate(notes["quiz"], 1):
                out += [f"{i}. {_one_line(q.get('q'))}",
                        f"   - Answer: {_one_line(q.get('a'))}"]
            out.append("")

    if tr:
        words = (tr.get("words") or {}).get("w") or []
        out += ["## Transcript", ""]
        para: list[str] = []
        start = None
        for s in tr.get("sents") or []:
            text = " ".join(words[s["w0"]: s["w0"] + s["wn"]])
            if not text:
                continue
            if start is None:
                start = s["t0"]
            para.append(text)
            if s["t1"] - start >= PARA_SECONDS * 1000:
                out += [f"**[{_clock(start)}]** " + " ".join(para), ""]
                para, start = [], None
        if para:
            out += [f"**[{_clock(start)}]** " + " ".join(para), ""]

    out += ["---", "*Exported from Lectern. Timestamps match the recording; "
            "notes written by a small language model can contain mistakes, "
            "so check anything important against the audio.*", ""]
    return "\n".join(out)
