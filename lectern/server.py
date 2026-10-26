"""Flask backend. Binds to 127.0.0.1 only.

The browser is a control surface: it holds no durable state, and it does not do
the recording. Every endpoint reads from disk, so a reloaded page (or a
completely restarted browser) reconstructs the full UI from server state.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

from . import audio, config, hardware, importer, models, storage, usage
from .pipeline import PIPELINE
from .recorder import RECORDER, RecorderBusy
from .storage import Lecture

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# .webmanifest is not in the stdlib table on macOS; without this it is served
# as text/plain and the browser refuses to install the app.
mimetypes.add_type("application/manifest+json", ".webmanifest")

app = Flask(__name__, static_folder=None)
app.json.sort_keys = False


COOKIE = "lectern_key"


_LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")
_LOOPBACK_HOSTS = {*_LOOPBACK_NAMES, *(f"{h}:{config.PORT}" for h in _LOOPBACK_NAMES)}


@app.before_request
def same_origin_only():
    """Stop other websites from driving this server.

    A page on any site can send a "simple" POST to 127.0.0.1 without a CORS
    preflight (e.g. start the microphone), and DNS rebinding can point an
    attacker's hostname at us to read lectures. So on loopback the Host must be
    ours, and anything that changes state must come from our own origin.
    """
    host = (request.host or "").lower()
    if config.is_loopback_only() and host not in _LOOPBACK_HOSTS:
        return Response("Forbidden host.\n", status=403, mimetype="text/plain")
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("Origin")
        if origin and origin.lower() not in (f"http://{host}", f"https://{host}"):
            return Response("Cross-site request refused.\n", status=403,
                            mimetype="text/plain")
        if not origin and request.headers.get("Sec-Fetch-Site") == "cross-site":
            return Response("Cross-site request refused.\n", status=403,
                            mimetype="text/plain")
    return None


@app.before_request
def require_token():
    """No-op on loopback. On a LAN bind, every request must carry the token."""
    if config.is_loopback_only():
        return None
    tok = config.access_token()
    given = (request.args.get("k") or request.headers.get("X-Lectern-Token")
             or request.cookies.get(COOKIE) or "")
    if secrets.compare_digest(given.encode(), tok.encode()):
        return None
    return Response(
        "Lectern is running in network mode. Open the link shown in the "
        "terminal, which includes your access key.\n", status=403,
        mimetype="text/plain")


@app.after_request
def set_token_cookie(resp):
    """Remember a valid ?k= so the phone does not need it on every request."""
    if not config.is_loopback_only():
        if request.args.get("k") and secrets.compare_digest(
                request.args.get("k").encode(), config.access_token().encode()):
            resp.set_cookie(COOKIE, config.access_token(), max_age=90 * 86400,
                            samesite="Lax", httponly=True)
    return resp


def _json_body() -> dict:
    """The JSON request body as a dict; anything else (an array, a string,
    invalid JSON) reads as empty instead of crashing the handler."""
    body = request.get_json(silent=True)
    return body if isinstance(body, dict) else {}


# ------------------------------------------------------------------ static
@app.get("/")
def index():
    resp = send_from_directory(WEB_DIR, "index.html")
    resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


@app.get("/web/manifest.webmanifest")
def manifest():
    """Served dynamically so start_url can carry the access key.

    iOS launches a home-screen app at the manifest's start_url, not at the URL
    you added it from, so a static "/" would drop the token and the app would
    open to a 403.
    """
    data = json.loads((WEB_DIR / "manifest.webmanifest").read_text())
    if not config.is_loopback_only():
        data["start_url"] = f"/?k={config.access_token()}"
    resp = Response(json.dumps(data, indent=2),
                    mimetype="application/manifest+json")
    resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


@app.get("/web/<path:filename>")
def web(filename: str):
    resp = send_from_directory(WEB_DIR, filename)
    # A cached ES module survives a reload and silently serves stale UI code
    # after an update. This is a localhost app; revalidation costs nothing.
    resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


# ------------------------------------------------------------- recording
@app.get("/api/status")
def api_status():
    st = RECORDER.status()
    st["free_bytes"] = storage.free_bytes()
    st["jobs"] = PIPELINE.jobs()
    # A first-run user has no speech model yet; the UI says so up front
    # instead of letting them record a whole lecture that cannot transcribe.
    st["needs_model"] = models.active_path() is None
    return jsonify(st)


@app.get("/api/devices")
def api_devices():
    try:
        devs = [{"index": d.index, "name": d.name} for d in audio.list_audio_devices()]
        chosen = audio.resolve_device()
        return jsonify({"devices": devs,
                        "selected": {"index": chosen.index, "name": chosen.name}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/preflight")
def api_preflight():
    try:
        d = audio.resolve_device()
        res = audio.probe(d)
        return jsonify({"ok": True, "device": d.name, **res})
    except audio.PreflightError as e:
        return jsonify({"ok": False, "kind": e.kind, "error": e.message,
                        "detail": e.detail}), 200
    except Exception as e:
        return jsonify({"ok": False, "kind": "device_error", "error": str(e)}), 200


@app.post("/api/record/start")
def api_record_start():
    body = _json_body()
    title = (body.get("title") or "").strip() or "Untitled lecture"
    course = (body.get("course") or "").strip()
    folder = (body.get("folder") or "").strip()
    try:
        # Under the move lock: a class rename must never snapshot the folder
        # while a lecture is being created inside it (it would carry the new
        # lecture's directory away from the writer that is about to use it).
        with _MOVE_LOCK:
            return jsonify(RECORDER.start(title, course, folder=folder))
    except RecorderBusy as e:
        return jsonify({"error": str(e), "kind": "busy"}), 409
    except audio.PreflightError as e:
        return jsonify({"error": e.message, "kind": e.kind,
                        "detail": e.detail}), 400
    except Exception as e:
        return jsonify({"error": str(e), "kind": "error"}), 500


@app.post("/api/import")
def api_import():
    """Import a downloaded lecture video or audio file.

    The body is the raw file (not multipart), so it streams straight to disk
    with no temporary copy; title and folder come in the query string. The
    browser sends it with XMLHttpRequest to show upload progress.
    """
    title = (request.args.get("title") or "").strip() or "Imported lecture"
    folder = (request.args.get("folder") or "").strip()
    filename = (request.args.get("filename") or "").strip()[:200]
    length = request.content_length
    if not length:
        return jsonify({"error": "empty upload"}), 400
    if storage.free_bytes() < length * 1.3 + config.MIN_FREE_BYTES_STOP:
        return jsonify({"error": "not enough free disk space for this file"}), 507
    with _MOVE_LOCK:
        lec = Lecture.create(title, "", folder, state="importing",
                             source={"kind": "import", "filename": filename})
    part = lec.work / (importer.SOURCE_NAME + ".part")
    got = 0
    try:
        with open(part, "wb") as f:
            while True:
                block = request.stream.read(1 << 20)
                if not block:
                    break
                f.write(block)
                got += len(block)
    except Exception as e:                 # disk error, or the client went away
        part.unlink(missing_ok=True)       # don't keep a multi-GB partial file
        lec.set_state("failed", failure="the upload did not finish; import the file again")
        return jsonify({"error": f"upload failed: {e}"}), 400
    if got != length:
        # Never hand a truncated file to the pipeline.
        part.unlink(missing_ok=True)
        lec.set_state("failed", failure="the upload did not finish; import the file again")
        return jsonify({"error": "the upload did not finish"}), 400
    src = importer.source_path(lec)
    part.replace(src)
    try:
        importer.probe(src)
    except Exception as e:
        # Unreadable file, or ffprobe missing: release the upload and leave a
        # failed (deletable) lecture, never one stuck in "importing".
        src.unlink(missing_ok=True)
        msg = str(e) if isinstance(e, importer.ImportError_) else f"could not read the file: {e}"
        lec.set_state("failed", failure=msg)
        return jsonify({"error": msg, "lecture_id": lec.id}), 415
    with _MOVE_LOCK:
        job = PIPELINE.submit_import(lec.id)
    return jsonify({"lecture_id": lec.id, "job": job})


@app.post("/api/record/stop")
def api_record_stop():
    try:
        res = RECORDER.stop()
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    with _MOVE_LOCK:                     # the pipeline must load the final path
        PIPELINE.submit(res["lecture_id"])
    return jsonify(res)


# ------------------------------------------------------------ legal
REPO_DIR = WEB_DIR.parent
LEGAL_DOCS = {"license": "LICENSE", "notices": "THIRD_PARTY_NOTICES.md",
              "privacy": "PRIVACY.md", "terms": "TERMS.md"}
# Bump when the first-run recording notice changes materially, so people
# see the new text once.
NOTICE_VERSION = "1"


def notice_accepted() -> bool:
    return models.load_settings().get("notice_accepted") == NOTICE_VERSION


@app.get("/api/legal/<doc>")
def api_legal(doc: str):
    name = LEGAL_DOCS.get(doc)
    if not name:
        abort(404)
    return send_from_directory(REPO_DIR, name, mimetype="text/plain; charset=utf-8")


@app.get("/api/notice")
def api_notice_get():
    return jsonify({"accepted": notice_accepted(), "version": NOTICE_VERSION})


@app.post("/api/notice")
def api_notice_accept():
    models.save_settings(notice_accepted=NOTICE_VERSION)
    return jsonify({"accepted": True, "version": NOTICE_VERSION})


def quick_title(now: datetime | None = None) -> str:
    """Auto title for a one-key recording: start date and time, e.g. "102626 2:05 PM"."""
    now = now or datetime.now()
    hour = now.hour % 12 or 12
    return f"{now:%m%d%y} {hour}:{now:%M} {'AM' if now.hour < 12 else 'PM'}"


@app.post("/api/record/toggle")
def api_record_toggle():
    """One key (or the menu bar) to start or stop. No dialog, no title prompt,
    no mic test: speed matters more here, and the helper still fails fast if
    the mic cannot open. Stopping goes straight to transcription."""
    if RECORDER.status().get("recording"):
        resp = api_record_stop()
        if isinstance(resp, tuple):
            return resp
        return jsonify({**resp.get_json(), "action": "stopped"})
    if not notice_accepted():
        # The menu bar can start a recording without the app ever being
        # opened; the recording-law notice must have been read first.
        return jsonify({"error": "Open Lectern once to read the recording notice first.",
                        "kind": "notice"}), 409
    try:
        with _MOVE_LOCK:
            st = RECORDER.start(quick_title(), "", folder="", skip_probe=True)
        return jsonify({**st, "action": "started"})
    except RecorderBusy as e:
        return jsonify({"error": str(e), "kind": "busy"}), 409
    except audio.PreflightError as e:
        return jsonify({"error": e.message, "kind": e.kind, "detail": e.detail}), 400
    except Exception as e:
        return jsonify({"error": str(e), "kind": "error"}), 500


_SHORTCUT_MODS = ("cmd", "ctrl", "opt", "shift")
DEFAULT_SHORTCUT = {"enabled": True, "key": "L", "mods": ["cmd"]}


@app.get("/api/shortcut")
def api_shortcut_get():
    """The global start/stop key, read by the menu bar helper."""
    return jsonify(models.load_settings().get("shortcut") or DEFAULT_SHORTCUT)


@app.post("/api/shortcut")
def api_shortcut_set():
    body = _json_body()
    key = str(body.get("key") or "").upper()
    mods = body.get("mods")
    if not (len(key) == 1 and key.isascii() and key.isalnum()):
        return jsonify({"error": "pick a letter or digit"}), 400
    if (not isinstance(mods, list) or not any(m != "shift" for m in mods)
            or any(m not in _SHORTCUT_MODS for m in mods)):
        return jsonify({"error": "hold at least one of ⌘ ⌃ ⌥ ⇧"}), 400
    sc = {"enabled": bool(body.get("enabled", True)), "key": key,
          "mods": [m for m in _SHORTCUT_MODS if m in mods]}
    models.save_settings(shortcut=sc)
    return jsonify(sc)


@app.post("/api/record/pause")
def api_record_pause():
    try:
        return jsonify(RECORDER.pause())
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/record/resume")
def api_record_resume():
    body = _json_body()
    lecture_id = (body.get("lecture_id") or "").strip()
    if not lecture_id:
        return jsonify({"error": "lecture_id is required"}), 400
    try:
        with _MOVE_LOCK:                 # no class move may land mid-spawn
            if PIPELINE.is_running(lecture_id):
                return jsonify({"error": "this lecture is still processing — "
                                         "wait for it to finish first",
                                "kind": "busy"}), 409
            return jsonify(RECORDER.resume(lecture_id))
    except RecorderBusy as e:
        return jsonify({"error": str(e), "kind": "busy"}), 409
    except audio.PreflightError as e:
        return jsonify({"error": e.message, "kind": e.kind,
                        "detail": e.detail}), 400
    except (FileNotFoundError, ValueError):
        return jsonify({"error": "no such lecture"}), 404
    except Exception as e:
        return jsonify({"error": str(e), "kind": "error"}), 500


# ------------------------------------------------- system and speech models
@app.get("/api/system")
def api_system():
    """Hardware, the speech models available, and which one will be used."""
    return jsonify(models.describe())


@app.post("/api/models/select")
def api_model_select():
    body = request.get_json(silent=True)
    mid = body.get("id") if isinstance(body, dict) else None
    if mid is not None and not isinstance(mid, str):
        return jsonify({"error": "unknown model"}), 404
    if mid is None:                       # clear the choice: back to automatic
        models.save_settings(whisper_model=None)
    elif mid not in models.BY_ID:
        return jsonify({"error": "unknown model"}), 404
    elif models.installed_path(models.BY_ID[mid]) is None:
        return jsonify({"error": "that model is not downloaded yet"}), 409
    else:
        models.save_settings(whisper_model=mid)
    return jsonify(models.describe())


@app.post("/api/models/download")
def api_model_download():
    """Download is always an explicit user action; Lectern never fetches on its own."""
    body = request.get_json(silent=True)
    mid = body.get("id") if isinstance(body, dict) else ""
    try:
        return jsonify(models.DOWNLOADER.start(mid if isinstance(mid, str) else ""))
    except KeyError:
        return jsonify({"error": "unknown model"}), 404


@app.post("/api/models/download/cancel")
def api_model_download_cancel():
    models.DOWNLOADER.cancel()
    return jsonify(models.DOWNLOADER.status())


@app.get("/api/storage")
def api_storage():
    """Disk used by Lectern (lectures, models, app) against the whole disk."""
    return jsonify(usage.report(force=request.args.get("fresh") == "1"))


@app.get("/api/diagnostics")
def api_diagnostics():
    """For the user to copy into a bug report. Never sent anywhere by the app."""
    d = models.describe()
    return jsonify({"app": f"{config.APP_NAME} {config.VERSION}",
                    "hardware": d["hardware"], "speech_model": d["active"],
                    "whisper_cli": config.WHISPER_CLI, "ffmpeg": config.FFMPEG})


# -------------------------------------------------------------- lectures
@app.get("/api/lectures")
def api_lectures():
    return jsonify({"lectures": [l.to_summary() for l in Lecture.all()]})


def _load(lecture_id: str) -> Lecture:
    try:
        return Lecture.load(lecture_id)
    except (FileNotFoundError, ValueError):
        abort(404)


@app.get("/api/lecture/<lecture_id>")
def api_lecture(lecture_id: str):
    lec = _load(lecture_id)
    d = lec.to_summary()
    d["job"] = PIPELINE.job(lecture_id)
    d["manifest"] = lec.manifest()
    return jsonify(d)


@app.get("/api/lecture/<lecture_id>/transcript")
def api_transcript(lecture_id: str):
    lec = _load(lecture_id)
    if not lec.transcript_path.is_file():
        return jsonify({"error": "no transcript yet"}), 404
    return send_file(lec.transcript_path, mimetype="application/json",
                     conditional=True, max_age=0)


@app.get("/api/lecture/<lecture_id>/notes")
def api_notes(lecture_id: str):
    lec = _load(lecture_id)
    p = lec.path / "notes.json"
    if not p.is_file():
        return jsonify({"error": "no notes yet"}), 404
    return Response(p.read_bytes(), mimetype="application/json")


@app.post("/api/lecture/<lecture_id>/notes/regenerate")
def api_notes_regen(lecture_id: str):
    lec = _load(lecture_id)
    with _MOVE_LOCK:
        if PIPELINE.is_running(lecture_id):
            return jsonify({"error": "already processing"}), 409
        return jsonify(PIPELINE.submit_notes(lecture_id))


@app.post("/api/lecture/<lecture_id>/process")
def api_process(lecture_id: str):
    """Transcribe and write notes. Also how a PAUSED lecture is finished
    without recording more ("Finish & transcribe")."""
    _load(lecture_id)
    with _MOVE_LOCK:
        if RECORDER.current_lecture_id() == lecture_id:
            return jsonify({"error": "this lecture is still recording — "
                                     "stop it first"}), 409
        return jsonify(PIPELINE.submit(lecture_id))


@app.delete("/api/lecture/<lecture_id>")
def api_delete(lecture_id: str):
    """Move a lecture to .trash -- never a direct delete. Restorable for 7
    days by moving it back out of ~/Lectures/.trash/ in Finder; the sweep on
    a later backend startup finishes the job automatically after that."""
    with _MOVE_LOCK:                     # trashing is a directory move too
        lec = _load(lecture_id)
        if RECORDER.current_lecture_id() == lecture_id:
            return jsonify({"error": "this lecture is currently recording — "
                                     "stop or pause it first"}), 409
        if PIPELINE.is_running(lecture_id) or lec.manifest().get("state") == "importing":
            return jsonify({"error": "this lecture is still processing — "
                                     "wait for it to finish first"}), 409
        dest = lec.trash()
    return jsonify({"deleted": True, "trash_path": dest.name})


# ------------------------------------------------------------- downloads
_DOWNLOAD_KINDS = {
    "audio": lambda lec: ((lec.playable_path() or lec.archive_path),
                          "audio/wav" if (lec.playable_path() and lec.playable_path().suffix == ".wav")
                          else "audio/mp4"),
    "transcript": lambda lec: (lec.transcript_txt_path, "text/plain"),
    "notes": lambda lec: (lec.notes_path, "text/markdown"),
}


@app.get("/api/lecture/<lecture_id>/download/<kind>")
def api_download(lecture_id: str, kind: str):
    """Plain-file downloads -- the same files that live in ~/Lectures, so you
    can save them, email them, or hand them to another tool (including
    pasting notes.md or transcript.txt into a chat) without opening Finder."""
    lec = _load(lecture_id)
    if kind == "markdown":
        from .export import build_markdown
        md = build_markdown(lec)
        if md is None:
            return jsonify({"error": "no notes or transcript for this lecture yet"}), 404
        slug = storage.slugify(lec.manifest().get("title") or lec.id)
        return Response(md, mimetype="text/markdown", headers={
            "Content-Disposition": f'attachment; filename="{slug}.md"'})
    getter = _DOWNLOAD_KINDS.get(kind)
    if getter is None:
        return jsonify({"error": "unknown download kind"}), 404
    path, mime = getter(lec)
    if not path.is_file():
        return jsonify({"error": f"no {kind} for this lecture yet"}), 404
    slug = storage.slugify(lec.manifest().get("title") or lec.id)
    return send_file(path, mimetype=mime, as_attachment=True,
                     download_name=f"{slug}{path.suffix}")


@app.patch("/api/lecture/<lecture_id>")
def api_patch(lecture_id: str):
    lec = _load(lecture_id)
    body = _json_body()
    fields = {}
    for k in ("title", "course"):
        if isinstance(body.get(k), str):
            fields[k] = body[k].strip()
    if fields:
        lec.update(**fields)          # dir name is immutable; only metadata moves
    return jsonify(lec.to_summary())


@app.get("/api/lecture/<lecture_id>/audio")
def api_audio(lecture_id: str):
    """Audio with HTTP Range support.

    conditional=True is what makes seeking work: without a 206 + Content-Range
    the browser cannot seek, which silently breaks click-to-jump.
    """
    lec = _load(lecture_id)
    path = lec.playable_path()
    if path is None:
        abort(404)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if path.suffix == ".m4a":
        mime = "audio/mp4"
    elif path.suffix == ".wav":
        mime = "audio/wav"
    resp = send_file(path, mimetype=mime, conditional=True,
                     max_age=0, last_modified=path.stat().st_mtime)
    resp.headers["Accept-Ranges"] = "bytes"
    return resp


# --------------------------------------------------------------- folders
@app.get("/api/folders")
def api_folders():
    return jsonify({"folders": storage.list_folders()})


@app.post("/api/folders")
def api_folder_create():
    body = _json_body()
    try:
        return jsonify({"name": storage.create_folder(body.get("name") or "")})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except OSError as e:
        return jsonify({"error": f"could not create folder: {e}"}), 400


@app.patch("/api/folders/<path:name>")
def api_folder_rename(name: str):
    body = _json_body()
    try:
        old = storage.clean_folder(name)
        with _MOVE_LOCK:
            # Renaming the class directory moves every lecture inside it, so
            # it has the same hazard as moving one lecture under a writer.
            if any(_busy(l.id) for l in Lecture.all() if l.folder == old):
                return jsonify({"error": f"a lecture in {old} is still recording "
                                         "or processing — rename the class once "
                                         "it finishes"}), 409
            new = storage.rename_folder(name, body.get("name") or "")
            for l in Lecture.all():            # queued moves follow the rename
                if l.pending_folder() == old:
                    l.set_pending_folder(new)
        return jsonify({"name": new})
    except FileNotFoundError:
        return jsonify({"error": "no such folder"}), 404
    except FileExistsError:
        return jsonify({"error": "a folder with that name already exists"}), 409
    except (ValueError, OSError) as e:
        return jsonify({"error": str(e)}), 400


@app.delete("/api/folders/<path:name>")
def api_folder_delete(name: str):
    try:
        storage.delete_folder(name)
        return jsonify({"deleted": True})
    except FileNotFoundError:
        return jsonify({"error": "no such folder"}), 404
    except OSError as e:
        return jsonify({"error": str(e)}), 409


@app.post("/api/lecture/<lecture_id>/folder")
def api_lecture_move(lecture_id: str):
    """File a lecture under a class. Immediate when the lecture is idle; while
    it is recording or processing the move is queued and applied as soon as
    nothing is writing into it (the response says "deferred")."""
    lec = _load(lecture_id)
    body = _json_body()
    folder = storage.clean_folder(body.get("folder") or "")
    try:
        with _MOVE_LOCK:
            lec = _load(lecture_id)      # fresh path: a queued move may have run
            if _busy(lec.id):
                lec.set_pending_folder(None if folder == lec.folder else folder)
                return jsonify({**lec.to_summary(), "deferred": True})
            lec.move_to_folder(folder)
    except FileExistsError:
        return jsonify({"error": "a lecture with that name is already there"}), 409
    except OSError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(lec.to_summary())


# --------------------------------------------------------- safe moves
log = logging.getLogger(__name__)

# Every directory move (a lecture, or a whole class folder) and every action
# that starts a writer on an existing lecture (resume, process, notes) takes
# this lock, so "is anything writing into it?" and "move it" are atomic.
_MOVE_LOCK = threading.Lock()
# Set once startup recovery has run. Until then a lecture that looks idle may
# be one recovery is about to salvage and requeue, so every move is queued.
_STARTUP_DONE = threading.Event()


def _busy(lecture_id: str) -> bool:
    """Something is, or may be about to start, writing into this lecture's
    directory: the recorder, the pipeline, startup recovery, or a lecture whose
    lecture.json still says "recording" with no writer known to this process
    (an orphan recovery has not reached, or the moment between a stop and its
    pipeline being queued)."""
    if (RECORDER.current_lecture_id() == lecture_id
            or PIPELINE.is_running(lecture_id)
            or not _STARTUP_DONE.is_set()):
        return True
    try:
        # "importing": the upload may still be streaming into .work.
        return Lecture.load(lecture_id).manifest().get("state") in ("recording", "importing")
    except (FileNotFoundError, ValueError):
        return False


def _apply_pending_moves() -> int:
    moved = 0
    for lec in Lecture.all():
        if lec.pending_folder() is None:
            continue
        with _MOVE_LOCK:
            # Re-read everything under the lock: the user may have moved it,
            # retargeted the queued move, or cancelled it since the scan.
            try:
                lec = Lecture.load(lec.id)
            except (FileNotFoundError, ValueError):
                continue
            target = lec.pending_folder()
            if target is None or _busy(lec.id):
                continue
            try:
                if target != lec.folder:
                    lec.move_to_folder(target)
                    moved += 1
                    log.info("filed %s under %r", lec.id, target or "Unfiled")
                lec.set_pending_folder(None)
            except FileExistsError:
                log.warning("cannot file %s under %r: name taken; dropping the "
                            "queued move", lec.id, target)
                lec.set_pending_folder(None)
            except OSError:
                log.exception("queued move of %s failed; will retry", lec.id)
    return moved


def _move_sweeper() -> None:
    _STARTUP_DONE.wait()
    while True:
        time.sleep(3.0)
        try:
            _apply_pending_moves()
        except Exception:
            log.exception("pending-move sweep failed")


def _startup_maintenance() -> None:
    """Recovery and the deferred WAV release. Never blocks the UI coming up."""
    from . import capture, recovery
    try:
        if capture.ensure_built() is None:
            app.logger.error("capture helper unavailable; recordings will use "
                             "ffmpeg, which drops audio")
    except Exception:
        app.logger.exception("capture helper build failed")
    try:
        ids = recovery.run_startup_recovery(PIPELINE.submit, PIPELINE.submit_import)
        if ids:
            app.logger.warning("recovered interrupted lectures: %s", ids)
    except Exception:
        app.logger.exception("startup recovery failed")
    finally:
        _STARTUP_DONE.set()              # queued class moves may now be applied
    try:
        res = recovery.sweep_trash()
        if res["files_removed"]:
            app.logger.info("released %d working files (%.1f MB)",
                            res["files_removed"], res["bytes_freed"] / 1e6)
    except Exception:
        app.logger.exception("trash sweep failed")


def create_app() -> Flask:
    config.ensure_dirs()
    adopted = RECORDER.adopt()
    if adopted:
        app.logger.warning("adopted in-progress recording: %s", adopted)
    threading.Thread(target=_startup_maintenance, daemon=True).start()
    threading.Thread(target=_move_sweeper, daemon=True).start()
    return app
