# Privacy

**Short version: Lectern collects nothing. Your recordings never leave your computer.**

- **No account, no analytics, no telemetry, no ads.** Lectern does not send
  data to its developer or anyone else.
- **Everything is processed on your device.** Recording, transcription
  (whisper.cpp) and study notes (Ollama) all run locally. Audio, transcripts
  and notes are saved as ordinary files in `~/Lectures` and are yours to keep,
  move or delete.
- **Network use is limited to what you ask for:**
  - downloading a speech model when you press **Download** (from Hugging Face);
  - **update checks in the Mac app**, once a day unless you turn them off in
    Settings ▸ Updates (details below);
  - links you click (GitHub, donation page), which open in your browser.
- **The app window is local too.** The Mac app shows Lectern in its own window,
  served by a small server on your computer that only accepts connections from
  this computer (127.0.0.1). The Windows beta opens the same local page in your
  browser.
- **Donations** go through Ko-fi in your browser. Ko-fi and its payment
  providers handle any payment details under their own privacy policies;
  Lectern never sees them.
- **Copy diagnostics** puts system details (computer model, memory, versions —
  never lecture content) on your clipboard. Nothing is sent; you decide whether
  to paste it into a bug report.
- **Phone mode** (`Lectern-on-Phone.command`) makes Lectern reachable from
  devices on your local network, protected by an access key. Only use it on a
  network you trust.
- **Recording other people** may be regulated where you are, and your school
  may have its own rules. See the recording notice in `TERMS.md`.

## Update checks (Mac app)

Once a day, and when you press **Check now**, the Mac app downloads one small
file, `appcast.xml`, from the Lectern website (lectern-ten.vercel.app) to see
whether a newer version exists. The request carries what any web request
carries: your IP address, plus the Lectern and Sparkle (the update tool)
version numbers. It contains no recordings, transcripts, lecture names, device
identifiers or system profile. If a newer version exists, the app tells you and
installs it only after you press Install; the update downloads from GitHub and
its signature is checked against a key built into the app before it is used.
Turn the daily check off in Settings ▸ Updates; **Check now** still works. The
Windows beta has no update check.

## The website

The Lectern website sets no cookies and runs no analytics or ads, and its fonts,
pictures and demo are served from the site itself. It is hosted by Vercel,
which, like any web host, processes your IP address and request details to
deliver pages and keep the service secure (see Vercel's privacy policy).
Downloads come from GitHub, which processes the request under GitHub's privacy
statement.

## Changes and contact

If this policy changes, the new version will be in this file with a new date.
Questions: [open an issue](https://github.com/lqoro/Lectern/issues).

Last updated 2026-10-30.
