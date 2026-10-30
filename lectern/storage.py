"""On-disk lecture storage. Plain files are the source of truth.

Durability notes verified on this machine:
  * fcntl.F_FULLFSYNC == 51 exists. On macOS, os.fsync() pushes data out of the
    page cache but does NOT flush the drive's write cache; only F_FULLFSYNC
    does. For "the laptop lost power", os.fsync alone is not a guarantee.
  * os.replace is atomic only within a filesystem, so every .part file is
    created in its FINAL directory via mkstemp(dir=...), never in /tmp.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from . import config
from . import platform_compat as pc

SCHEMA = 1
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")

# Every read-modify-write of a lecture.json goes through this lock, so a
# rename from the UI can't drop fields the pipeline just wrote (or vice versa).
_MANIFEST_LOCK = threading.RLock()

# Ordered lifecycle. `state` in lecture.json is always one of these.
STATES = ("recording", "importing", "paused", "interrupted", "concatenating", "transcribing",
          "noting", "archiving", "complete", "failed")
TERMINAL = ("complete", "failed")


# ---------------------------------------------------------------- atomics
def atomic_write_bytes(path: Path, data: bytes, *, durable: bool = False,
                       mkdirs: bool = True) -> None:
    """mkdirs=False: fail (FileNotFoundError) rather than recreate a missing
    parent -- a lecture moved to another class must not reappear as a ghost
    directory holding only a lecture.json written by a stale Lecture object."""
    path = Path(path)
    d = path.parent
    if mkdirs:
        d.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(d), prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb", closefd=False) as f:
            f.write(data)
            f.flush()
        if durable:
            pc.fsync_fd(fd)                      # F_FULLFSYNC on macOS — see docstring
    except BaseException:
        os.close(fd)
        os.unlink(tmp)
        raise
    else:
        os.close(fd)
    os.chmod(tmp, 0o644)      # mkstemp defaults to 0600; these are user documents
    replace(tmp, path)
    if durable:
        pc.fsync_dir(d)                          # persist the directory entry


def replace(src, dst) -> None:
    """os.replace, retried briefly on Windows: there a file (or a folder holding
    one) that another thread, the player, antivirus or the indexer has open
    cannot be replaced or renamed for a moment (WinError 5/32)."""
    for attempt in range(20 if pc.IS_WIN else 1):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == (19 if pc.IS_WIN else 0):
                raise
            time.sleep(0.05)


def atomic_write_text(path: Path, text: str, *, durable: bool = False,
                      mkdirs: bool = True) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), durable=durable, mkdirs=mkdirs)


def atomic_write_json(path: Path, obj, *, durable: bool = False, indent=None,
                      mkdirs: bool = True) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent),
                      durable=durable, mkdirs=mkdirs)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def boot_seconds() -> int:
    """Boot time as a PID-reuse-proof liveness token.

    If the machine rebooted, any PID recorded before is meaningless — which is
    exactly the power-loss case we must detect.
    """
    return pc.boot_seconds()


# Every id is created_at + slugify() (+ "-n"): letters, digits, "_" and "-".
_LECTURE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def slugify(title: str, maxlen: int = 48) -> str:
    s = unicodedata.normalize("NFKD", title or "").encode("ascii", "ignore").decode()
    s = _SLUG_STRIP.sub("-", s.lower()).strip("-")
    return (s[:maxlen].strip("-")) or "lecture"


# ---------------------------------------------------------------- lecture
class Lecture:
    """One lecture directory. The directory name is the immutable id."""

    def __init__(self, path: Path):
        self.path = Path(path)

    # -- identity
    @property
    def id(self) -> str:
        return self.path.name

    @property
    def work(self) -> Path:
        return self.path / ".work"

    @property
    def manifest_path(self) -> Path:
        return self.path / "lecture.json"

    @property
    def transcript_path(self) -> Path:
        return self.path / "transcript.json"

    @property
    def transcript_txt_path(self) -> Path:
        return self.path / "transcript.txt"

    @property
    def notes_path(self) -> Path:
        return self.path / "notes.md"

    @property
    def master_wav(self) -> Path:
        return self.work / "master.wav"

    @property
    def archive_path(self) -> Path:
        return self.path / f"audio.{config.ARCHIVE_EXT}"

    def segments(self) -> list[Path]:
        if not self.work.is_dir():
            return []
        return sorted(p for p in self.work.glob("seg-*.wav") if p.stat().st_size > 44)

    # -- folders
    @property
    def folder(self) -> str:
        """Class folder name, or "" for an unfiled lecture."""
        parent = self.path.parent
        return "" if parent == config.LECTURES_ROOT else parent.name

    # A move asked for while the lecture is recording or processing is parked
    # in this sidecar and applied once it is idle (server._apply_pending_moves).
    # Renaming the directory under a running writer would send the next
    # segment -- or the pipeline's next file -- to a path that no longer
    # exists. It is a separate file, not a lecture.json field, because the
    # recorder and pipeline rewrite lecture.json read-modify-write and would
    # silently drop it.
    @property
    def pending_move_path(self) -> Path:
        return self.path / ".move-to"

    def pending_folder(self) -> str | None:
        try:
            return clean_folder(self.pending_move_path.read_text())
        except OSError:
            return None

    def set_pending_folder(self, folder: str | None) -> None:
        if folder is None:
            try:
                self.pending_move_path.unlink()
            except FileNotFoundError:
                pass
            return
        atomic_write_text(self.pending_move_path, clean_folder(folder), mkdirs=False)

    def move_to_folder(self, folder: str) -> "Lecture":
        """Move this lecture into a class folder. The id never changes.
        Callers must make sure nothing is writing into it (see server)."""
        folder = clean_folder(folder)
        dest_dir = (config.LECTURES_ROOT / folder) if folder else config.LECTURES_ROOT
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / self.path.name
        if dest == self.path:
            self.set_pending_folder(None)   # already there: nothing left to apply
            return self
        if dest.exists():
            raise FileExistsError(f"{dest} already exists")
        self.path.rename(dest)          # same filesystem: atomic
        self.path = dest
        self.set_pending_folder(None)
        return self

    # -- creation
    @classmethod
    def create(cls, title: str, course: str = "", folder: str = "", *,
               state: str = "recording", source: dict | None = None) -> "Lecture":
        config.ensure_dirs()
        now = datetime.now()
        base = f"{now:%Y-%m-%d_%H%M}_{slugify(title)}"
        folder = clean_folder(folder)
        root = (config.LECTURES_ROOT / folder) if folder else config.LECTURES_ROOT
        root.mkdir(parents=True, exist_ok=True)
        # The id (directory name) must be unique across ALL class folders,
        # because Lecture.load finds a lecture by id alone.
        taken = {p.name for p in config.LECTURES_ROOT.iterdir()}
        for d in _folder_dirs():
            taken.update(p.name for p in d.iterdir())
        name, n = base, 2
        while name in taken:
            name = f"{base}-{n}"
            n += 1
        path = root / name
        path.mkdir(parents=True)
        (path / ".work").mkdir()
        (path / ".work" / ".metadata_never_index").touch()
        lec = cls(path)
        lec.write_manifest({
            "schema": SCHEMA,
            "id": lec.id,
            "title": title or "Untitled lecture",
            "course": course,
            "created_utc": utcnow(),
            "state": state,
            "stage_history": [{"s": state, "at": utcnow()}],
            "source": source or {"kind": "microphone"},
            "recovered": False,
            "recovery_note": None,
            "audio": {}, "transcript": {}, "notes": {},
            "app": {"name": config.APP_NAME, "version": config.VERSION},
        }, durable=True)
        lec.set_stage(state)
        return lec

    @classmethod
    def load(cls, lecture_id: str) -> "Lecture":
        # Refuse anything that could escape the root.
        # Real ids are slugify() output plus "-n": letters, digits, "_" and "-".
        # Anything else (a backslash or drive letter on Windows) could escape.
        if not _LECTURE_ID.fullmatch(lecture_id):
            raise ValueError(f"invalid lecture id: {lecture_id!r}")
        # The id is the directory basename and stays constant when a lecture is
        # filed into a class folder, so look at the root and one level down.
        direct = config.LECTURES_ROOT / lecture_id
        if (direct / "lecture.json").is_file():
            return cls(direct)
        for folder in _folder_dirs():
            p = folder / lecture_id
            if (p / "lecture.json").is_file():
                return cls(p)
        raise FileNotFoundError(f"no such lecture: {lecture_id}")

    def trash(self) -> Path:
        """Move this lecture into .trash. Restorable for 7 days; never a
        direct delete. The 7-day sweep in recovery.py finishes the job later,
        in a different process, the same way archived .work dirs are handled.
        """
        config.ensure_dirs()
        dest = config.TRASH_DIR / self.path.name
        n = 2
        while dest.exists():
            dest = config.TRASH_DIR / f"{self.path.name}-{n}"
            n += 1
        replace(self.path, dest)
        # rename keeps the old mtime, and the 7-day sweep ages entries by
        # mtime: without this a 3-week-old lecture is purged on next start.
        now = time.time()
        os.utime(dest, (now, now))
        self.path = dest
        return dest

    @classmethod
    def all(cls) -> list["Lecture"]:
        config.ensure_dirs()
        out = []
        for p in config.LECTURES_ROOT.glob("*/lecture.json"):
            if not p.parent.name.startswith("."):
                out.append(cls(p.parent))
        for folder in _folder_dirs():
            for p in folder.glob("*/lecture.json"):
                if not p.parent.name.startswith("."):
                    out.append(cls(p.parent))
        return sorted(out, key=lambda l: l.id, reverse=True)

    # -- manifest
    def manifest(self) -> dict:
        """For display: a missing or unreadable manifest reads as empty."""
        try:
            return self._manifest_strict()
        except (OSError, json.JSONDecodeError):
            return {}

    def _manifest_strict(self) -> dict:
        """For read-modify-write: a read error must raise, or the write that
        follows would replace the title, class and audio fields with nothing."""
        return json.loads(self.manifest_path.read_text())

    def write_manifest(self, m: dict, *, durable: bool = True) -> None:
        atomic_write_json(self.manifest_path, m, durable=durable, indent=2, mkdirs=False)

    def update(self, **fields) -> dict:
        with _MANIFEST_LOCK:
            m = self._manifest_strict()
            m.update(fields)
            self.write_manifest(m)
            return m

    def set_state(self, state: str, **extra) -> dict:
        assert state in STATES, state
        with _MANIFEST_LOCK:
            m = self._manifest_strict()
            m["state"] = state
            m.setdefault("stage_history", []).append({"s": state, "at": utcnow()})
            m.update(extra)
            self.write_manifest(m)
        self.set_stage(state)
        return m

    # -- resumable stage marker (filesystem-authoritative)
    def set_stage(self, stage: str) -> None:
        if self.work.is_dir():
            atomic_write_text(self.work / "stage", stage, durable=True, mkdirs=False)

    def stage(self) -> str:
        p = self.work / "stage"
        try:
            return p.read_text().strip()
        except OSError:
            return ""

    # -- derived
    def duration_ms(self) -> int:
        return int(self.manifest().get("audio", {}).get("duration_ms") or 0)

    def has_audio(self) -> bool:
        return self.archive_path.is_file() or self.master_wav.is_file()

    def playable_path(self) -> Path | None:
        """Archive if finalized, else the transient master WAV (M1 ships on WAV)."""
        if self.archive_path.is_file():
            return self.archive_path
        if self.master_wav.is_file():
            return self.master_wav
        return None

    def to_summary(self) -> dict:
        m = self.manifest()
        audio = m.get("audio", {}) or {}
        pl = self.playable_path()
        return {
            "id": self.id,
            "title": m.get("title") or self.id,
            "course": m.get("course") or "",
            "folder": self.folder,
            "pending_folder": self.pending_folder(),
            "created_utc": m.get("created_utc"),
            "state": m.get("state"),
            "recovered": bool(m.get("recovered")),
            "recovery_note": m.get("recovery_note"),
            "pause_reason": m.get("pause_reason") if m.get("state") == "paused" else None,
            "paused_utc": m.get("paused_utc") if m.get("state") == "paused" else None,
            "duration_ms": int(audio.get("duration_ms") or 0),
            "has_audio": pl is not None,
            "has_transcript": self.transcript_path.is_file(),
            "has_notes": self.notes_path.is_file(),
            "words": (m.get("transcript", {}) or {}).get("words") or 0,
            "bytes": (pl.stat().st_size if pl else 0),
        }


# --------------------------------------------------------------- folders
# Path separators, characters Windows forbids in names, control characters and
# lone surrogates (which cannot be encoded).
_FOLDER_BAD = re.compile(r'[/\\:<>"|?*\x00-\x1f\ud800-\udfff]')
# Windows device names: "CON" or "com1.txt" cannot be a folder there.
_RESERVED = re.compile(r"(CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(\..*)?", re.I)
FOLDER_MAX = 60


def clean_folder(name: str) -> str:
    """Normalise a class-folder name, or "" for unfiled.

    Folders are real directories under ~/Lectures, so the name has to be safe
    as a path component and must never traverse upward.
    """
    name = _FOLDER_BAD.sub("", (name or "").strip()).strip(". ")
    if name in ("", ".", ".."):
        return ""
    name = name[:FOLDER_MAX].strip(". ")
    return name + "_" if _RESERVED.fullmatch(name) else name


def _folder_dirs() -> list[Path]:
    """Immediate subdirectories of ~/Lectures that hold lectures, not one."""
    config.ensure_dirs()
    out = []
    for p in config.LECTURES_ROOT.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        if (p / "lecture.json").is_file():
            continue                       # that is a lecture, not a folder
        out.append(p)
    return sorted(out, key=lambda p: p.name.lower())


def list_folders() -> list[dict]:
    counts: dict[str, int] = {}
    for lec in Lecture.all():
        counts[lec.folder] = counts.get(lec.folder, 0) + 1
    out = [{"name": p.name, "count": counts.get(p.name, 0)} for p in _folder_dirs()]
    if counts.get("", 0):
        out.insert(0, {"name": "", "count": counts[""]})
    return out


def create_folder(name: str) -> str:
    name = clean_folder(name)
    if not name:
        raise ValueError("folder name is empty after cleaning")
    # "bio" when "Bio" exists is the same folder on a case-insensitive disk
    # (macOS, Windows): use the name that is really there.
    same = next((p.name for p in _folder_dirs() if p.name.lower() == name.lower()), None)
    if same:
        return same
    (config.LECTURES_ROOT / name).mkdir(parents=True, exist_ok=True)
    return name


def rename_folder(old: str, new: str) -> str:
    old_c, new_c = clean_folder(old), clean_folder(new)
    if not old_c or not new_c:
        raise ValueError("folder names must be non-empty")
    src = config.LECTURES_ROOT / old_c
    dst = config.LECTURES_ROOT / new_c
    if not src.is_dir():
        raise FileNotFoundError(old_c)
    if dst.exists() and not os.path.samefile(src, dst):   # "bio" -> "Bio" is fine
        raise FileExistsError(new_c)
    replace(src, dst)
    return new_c


def free_bytes() -> int:
    return shutil.disk_usage(config.LECTURES_ROOT).free
