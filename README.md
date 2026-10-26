# Lectern

On-device lecture recorder, transcriber and click-to-jump player.
Everything runs locally on your Mac — no cloud, no API keys, no account.
Free and open source (MIT).

**Status (v0.1.0):** tested on Apple Silicon Macs. Windows and Linux are
experimental (see below). It is a local web app you start from a script, not a
signed `.app` yet.

## Install (macOS)

You need [Homebrew](https://brew.sh). Then:

```bash
git clone https://github.com/lqoro/Lectern.git
cd Lectern
./install.sh            # whisper.cpp, ffmpeg, Python env
./install.sh --notes    # optional: AI study notes via Ollama (~2 GB, best with 8 GB+ RAM)
```

Or download the source zip from the
[latest release](https://github.com/lqoro/Lectern/releases/latest), unzip it,
and run `./install.sh` inside the folder. If macOS blocks a script, right-click
it ▸ **Open**.

On first launch open **Settings (⚙)** and download the speech model it
recommends for your computer (32 MB – 1 GB). Nothing is downloaded until you
press that button. The microphone helper is compiled on first start with the
Xcode Command Line Tools that Homebrew installs; macOS will ask for microphone
permission the first time you record.

## Quick record from anywhere (menu bar + shortcut)

```bash
./install.sh --menubar
```

adds a mic icon to the menu bar and a global shortcut, **⌘L** by default. Press it
anywhere — Lectern doesn't need to be open — and a recording starts at once, titled
with the date and time (`102626 2:20 PM`). Press it again to stop and transcribe.
The menu also has Pause / Continue. Change or turn off the shortcut in
**Settings (⚙) ▸ Quick record**. Note that ⌘L normally jumps to a browser's address
bar; while the helper runs, it records instead, so pick another combination if you
use that a lot. Remove it with `.venv/bin/python -m lectern.menubar uninstall`.

## Run it

```bash
./Lectern.command
```

Double-clicking that file in Finder works too. It starts the backend if it isn't
already running and opens the app.

**Put it in your Dock** (opens instantly, own icon, no browser chrome):
open <http://127.0.0.1:8477/> in Safari, then **File ▸ Add to Dock**.
Chrome/Edge: **⋮ ▸ Cast, save and share ▸ Install page as app**.

**On your iPhone**, same wifi:

```bash
./Lectern-on-Phone.command
```

It prints a link containing an access key. Open it on the phone, then
**Share ▸ Add to Home Screen**. Treat that link like a password — while phone
mode is on, anything on the network can reach the port, so it is token-gated and
loopback-only by default.

## What it does today

- Records from the built-in microphone, straight to disk, in 60-second chunks,
  through a small native helper (`lectern_capture.swift`, built automatically on
  first start) that keeps every sample.
- Transcribes on the M1 GPU with whisper.cpp at roughly **7× real time** —
  a 50-minute lecture takes about 7 minutes.
- **Click any word in the transcript and the audio jumps to that moment.**
  Measured accuracy: 0 ms (playback starts 150 ms early on purpose, so you do
  not miss the first syllable of the word you clicked).
- Karaoke-style highlighting follows playback.
- **Class folders.** Real directories under `~/Lectures/BIO 201/…`, so they are
  the same folders you see in Finder. Pick the class first when you start
  recording (it is guessed from the title — "biology 102126" files under
  BIO — and "＋ New class…" makes one on the spot). Move a lecture later from
  its card (click the class tag, or ⋯ ▸ Move to class) or the Class menu on the
  lecture page. A move asked for while it is still recording or processing is
  queued and happens as soon as nothing is writing into it.
- **Closing the laptop pauses the recording and turns the mic off.** It stays
  paused when you reopen it: **Continue recording** picks up in the same
  lecture, **Finish & transcribe** processes what you have.
- **Compresses to AAC** — 14.6 MB per lecture-hour, about 0.7 GB a semester —
  but only after an 8-step verification gate passes.
- **Writes study notes** with a local `qwen2.5:3b`, plus key terms and
  exam-style questions. **Clicking a note jumps the audio to where that point
  was made**, even though the note is a paraphrase that appears nowhere
  verbatim in the transcript.

## Online lectures: import a recording

Click **Import** and pick a downloaded lecture video or audio file (Panopto,
Zoom or Canvas recordings, mp4/mov/webm/m4a/mp3/wav, phone voice memos). Lectern
pulls out the sound and runs the same transcript, notes and click-to-jump
pipeline; nothing is played back, so a one-hour video takes minutes, headphones
or not. Only the audio is kept (the uploaded file is released with the working
files once the archive verifies). Several files can be imported at once and are
processed in order. Check your school's or platform's policy before downloading
course videos.

## Speech models and weak computers

Lectern picks a speech model to suit the machine (`GET /api/system`): the
default `large-v3-turbo-q5_0` on Apple Silicon with 8 GB or more, `small.en` /
`base.en` / `tiny.en` on CPU-only or low-memory machines. Models are looked up in
`<support dir>/models/`, then the legacy `~/.localflow/models/`. Download one
with `POST /api/models/download {"id": "base.en"}` (only ever on request) and
choose it with `POST /api/models/select`. The CPU-only choices are heuristics,
not benchmarks.

Whisper and the AAC encode run below normal priority so a lecture being
recorded is never starved. If the notes model would not fit (under 5.5 GB RAM)
or is too slow (`LECTERN_NOTES_BUDGET_S`, default 30 min), the notes fall back
to verbatim key sentences instead of failing. Environment overrides:
`LECTERN_MODEL` (a file), `LECTERN_MODEL_DIR`, `LECTERN_THREADS`,
`LECTERN_NOTES=auto|llm|extractive`.

## Windows and Linux (experimental)

macOS keeps using its own Swift recorder. On Windows and Linux Lectern records
through a portable Python helper (`lectern/capture_py.py`, built on PortAudio),
which writes the same sample-exact 60-second segments. It is tested here with a
simulated microphone (`python -m unittest discover -s tests`) but has NOT yet
been run against real Windows/Linux hardware.

```
pip install -r requirements.txt      # flask, waitress, sounddevice, numpy, psutil
python -m lectern                    # or double-click Lectern.bat on Windows
```

You also need `ffmpeg` and `whisper-cli` (whisper.cpp) on your PATH, or set
`LECTERN_FFMPEG` / `LECTERN_WHISPER` / `LECTERN_BIN_DIR`. Set
`LECTERN_CAPTURE=python` to try the portable recorder on a Mac.

## Your recording cannot be lost by closing the browser

The browser is only a remote control. `ffmpeg` does the recording in its own
process session, so:

- closing the tab, or quitting the browser entirely, does nothing to it;
- if the backend crashes, the recording **keeps going**, and a restarted backend
  re-adopts it (verified: the backend was SIGKILLed mid-recording and the audio
  file kept growing);
- audio is flushed to disk every packet, so even a hard power cut costs at most
  a fraction of a second. A WAV truncated mid-write still transcribes correctly;
  verified by SIGKILLing ffmpeg mid-sentence, after which the salvaged 9.8 s
  still transcribed cleanly and the lecture was marked "recovered" with a note
  saying exactly what was lost.

The original recordings are never deleted on the strength of one check. After
the AAC archive is written it must pass a full decode of every packet, a
duration match against the true PCM frame count, a checksum and an atomic
rename; the working files are then **moved to `.trash`, not deleted**, and are
only unlinked on a later startup, in a different process, after the archive
verifies a second time.

Your data lives in `~/Lectures/` as plain files. See the README in that folder.

## Layout

```
lectern/
├── Lectern.command      double-click launcher
├── install.sh           one-time setup (macOS)
├── lectern/
│   ├── config.py        every path and tunable
│   ├── audio.py         device resolution by name + mic preflight
│   ├── lectern_capture.swift  native mic capture (replaces ffmpeg avfoundation)
│   ├── capture.py       builds/locates that helper
│   ├── capture_py.py    portable recorder (Windows/Linux; PortAudio)
│   ├── platform_compat.py  every macOS/Windows/Linux difference in one place
│   ├── export.py        one-file Markdown export (notes + transcript)
│   ├── pcm.py           tiny PCM helpers (replaces removed `audioop`)
│   ├── recorder.py      supervised capture, survives backend death
│   ├── concat.py        frame-count WAV assembly (never trusts a header)
│   ├── transcribe.py    whisper.cpp + normalisation to the word contract
│   ├── hardware.py      RAM / cores / accelerator, whisper thread count
│   ├── models.py        speech-model catalogue, choice, resumable download
│   ├── notes.py         study notes via ollama + extractive fallback
│   ├── anchor.py        TF-IDF grounding of notes back to timestamps
│   ├── archive.py       AAC encode + the verification gate
│   ├── recovery.py      startup salvage + deferred WAV release
│   ├── pipeline.py      stage lock enforcing the RAM budget
│   ├── menubar.py       builds/installs the menu bar helper
│   ├── lectern_menubar.swift  menu bar icon + global start/stop shortcut
│   └── server.py        Flask API, HTTP Range for audio seeking
└── web/                 no build step; plain ES modules
```

## Notes for whoever works on this next

Several non-obvious things were measured on an M1 MacBook, and the code depends
on them. Changing any of these will break something quietly:

- **Never record with `ffmpeg -f avfoundation`.** Its audio input holds a
  single pending buffer and overwrites it if ffmpeg has not polled in time
  (it sleeps 10 ms). Measured on macOS 27: ~88% of samples kept from a shell,
  **~32% when spawned detached by the backend** — every lecture from mid-Sep
  to 2026-09-28 lost about two thirds of its speech, and nothing noticed
  because segments still closed every 60 s of wall clock. The Swift helper
  writes every buffer and cuts segments by sample count; `recorder.py` falls
  back to ffmpeg only if the helper cannot be built. Check: a 60 s segment
  must be exactly 1,920,044 bytes.
- **The notes prompt must contain no subject-matter examples.** It used to
  show glycolysis/ATP examples; given a near-empty transcript, qwen2.5:3b
  copied them into history-lecture notes. Notes, terms, quiz and summary are now
  dropped unless they are made of words the lecture used, and a recording
  with under 25 distinct content words gets no notes at all.
- **whisper runs with `-mc 0 -sns`.** With previous-text context, one bad
  window poisons the rest: measured, the default produced one sentence 291
  times from a damaged lecture where `-mc 0 -sns` produced 48 real ones.

- (ffmpeg fallback only) **`-flush_packets 1` does nothing with ffmpeg's segment muxer.** It must be
  `-segment_format_options flush_packets=1`, which is passed through to the child
  WAV muxer. Measured: with the wrong flag, the in-progress file stays at 0 bytes
  for 8.192 s and then jumps by 262144 bytes.
- **`-dtw` is dead** in whisper.cpp 1.9.1 because flash attention is on by
  default and silently disables it. Every `t_dtw` is `-1`. Word timestamps come
  from each token's `offsets`.
- **`-ojf` alone already emits per-token timestamps**, so `-ml 1 -sow` is not
  needed and would destroy sentence structure.
- **`-fflags +bitexact` must be at the output position** or the WAV header is
  78 bytes instead of 44.
- **Audio device indices shift.** A Continuity iPhone mic sits at index 1 and
  moves everything below it. Always resolve by name immediately before spawning.
- **Whisper narrates silence** as `*music*` / `(coughing)`. These are filtered
  after word assembly, not per token — whisper splits `*music*` into `*`,
  `music`, `*`.
- Audio is served with `conditional=True` so the browser gets `206 Partial
  Content`. Without Range support, seeking — and therefore click-to-jump —
  silently breaks.
- **AAC, not Opus.** macOS 26 AVFoundation over-reports Ogg Opus duration by
  7.3%, which breaks the seek bar in Safari. AAC costs ~1.5 GB more across a
  four-year degree and removes the whole risk class.
- **`-f ipod` is passed explicitly** when encoding. The output is written to
  `audio.m4a.part` first, and ffmpeg cannot infer a muxer from `.part`.
- **A folder is a real directory**, and a lecture's id is its directory
  basename, which never changes when it is filed. `Lecture.load` therefore
  looks at the root and one level down.
- **Whisper falls into repetition loops on trailing silence.** Measured: one
  sentence emitted 20+ times across the last 25 s of a recording, which then
  poisoned the notes. Runs of three or more identical sentences are collapsed
  after word assembly.
- **IDF is computed over sentences, not the 1–3 sentence windows.** The windows
  overlap, so counting document frequency across them inflates every term ~6x
  and prunes ordinary vocabulary. The df ceiling is also disabled below 30
  sentences, where stopword statistics are meaningless.
- `num_ctx` is **constant across every ollama call**. Changing it mid-pipeline
  evicts and reloads the runner.
- **Closing the lid used to leave the mic on.** `caffeinate -i` only blocks
  idle sleep, so lid-close slept the Mac with the writer frozen mid-recording,
  and on wake it simply carried on — mic live, battery draining. Now: the
  helper hears will-sleep / clamshell-closed from IOKit, closes the segment,
  releases the mic, exits 75 and only then acks the sleep; the watcher also
  polls `AppleClamshellState` and detects a sleep gap (`CLOCK_MONOTONIC` minus
  `CLOCK_UPTIME_RAW`). Any writer exit we did not ask for leaves the lecture
  "paused" with a reason, never stuck in "recording" (running out of disk
  pauses too, it does not silently stop). Each watcher thread supervises only
  the session it was started for. `kill -USR1` on the helper exercises the lid
  path without closing the laptop.
- **Never rename a lecture or class directory under a live writer.** The next
  segment, or the pipeline's next file, would go to a path that no longer
  exists. Moves go through `server._MOVE_LOCK` + `_busy()`; a busy lecture
  gets a `.move-to` sidecar (not a lecture.json field — recorder and pipeline
  rewrite lecture.json read-modify-write) that a sweeper applies once idle.
  Until startup recovery has run, every lecture counts as busy, and so does
  one whose lecture.json still says "recording". A writer that paused itself
  for the lid while no backend was running is kept "paused" by recovery, not
  salvaged as a crash. `tests/test_folders_lid.py` covers all of this without
  a microphone.
- There is deliberately **no `shutil.rmtree` anywhere**. Deletion is an explicit
  allowlist of filenames, non-recursive, symlink-refusing, inside a directory
  whose full ancestry is asserted.

## How notes get anchored, and how much to trust them

A generated note is a paraphrase, so anchoring cannot be string search. Three
tiers, and **the UI shows which one you are looking at**:

| Tier | How | Precision |
|---|---|---|
| exact | you clicked a word in the transcript | the word itself |
| precise | the model cited a sentence *and* a TF-IDF check agrees | that sentence |
| region | everything else — the note inherits its ~5-minute window | a few minutes |

A region anchor is drawn dimmed with a `~`. It is deliberately coarse rather
than confidently wrong.

Two guards sit on top. `cites` is constrained by JSON Schema to an **enum of the
sentence ids actually present in that chunk**, so a hallucinated citation is
impossible at the token level rather than cleaned up afterwards. And a
*groundedness* score measures how much of a note's vocabulary appears anywhere
in the lecture; a note that scores low is marked with a `?` and a tooltip,
because a 3B model will occasionally supply a true fact from its own training
that your lecturer never said.

**Be sceptical of the notes, not of the timestamps.** The anchors were measured
correct; the notes are written by a 3 billion parameter model and are a study
aid, not a source of truth. Click through to the audio when a detail matters.
`notes.md` is yours to edit, and edits are picked up automatically.

## Still to come

- **M4** — full-text search across every lecture, and autostart at login.

## Help, bugs and ideas

- **Bug?** [Open a bug report](https://github.com/lqoro/Lectern/issues/new?template=bug.yml).
  Settings (⚙) ▸ **Copy diagnostics** puts system details (no lecture content)
  on your clipboard to paste in. Nothing is ever sent automatically.
- **Idea?** [Request a feature](https://github.com/lqoro/Lectern/issues/new?template=feature.yml).
- Pull requests welcome. Run the tests with `python -m unittest discover -s tests`.

## Third-party software

Lectern calls these tools; they are installed separately and keep their own
licences: [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and the Whisper
model weights (MIT), [FFmpeg](https://ffmpeg.org) (LGPL/GPL, via Homebrew),
[Ollama](https://ollama.com) (MIT, optional), Flask and Waitress (BSD/ZPL).
