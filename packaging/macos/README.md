# Building Lectern.app

`build_app.py` turns this repo into a normal Mac app: one `Lectern.app` (and a
`Lectern-<version>.dmg`) with the server, whisper, ffmpeg and the mic helper
inside. Someone who downloads it needs no Homebrew, Terminal or Xcode.

## Build it (no Apple developer account needed)

On an Apple Silicon Mac with this repo's `.venv` set up (`./install.sh`):

```bash
brew install cmake
.venv/bin/python packaging/macos/build_app.py
```

The first build compiles whisper.cpp and FFmpeg from pinned sources and takes
about 15 minutes; later builds reuse them from `build/macos/`. Output lands in
`dist/`. The script finishes with a smoke test that starts the bundled server
against throwaway folders (never `~/Lectures`) and checks the web UI, the legal
pages, whisper-cli and the AAC encoder.

## Try it on your own Mac

1. Quit any Lectern server that is already running (the app reuses whatever
   answers on port 8477, so you would be testing the old one), and remove the
   old menu bar helper so the two don't fight over ⌘L:
   `.venv/bin/python -m lectern.menubar uninstall`
2. Open `dist/Lectern-0.2.0.dmg` and drag Lectern to Applications.
3. Open Lectern. It should open in your browser, show the mic icon in the menu
   bar, and ask for the microphone the first time you record.
4. Download a speech model in Settings, record a short lecture, check the
   transcript and click-to-jump, then try ⌘L from another app.

Because this build is signed ad hoc, it runs on the Mac that built it, and
macOS may ask for the microphone again after each rebuild. Other Macs will
refuse to open it until the next two steps are done.

## After joining the Apple Developer Program

1. In Xcode ▸ Settings ▸ Accounts, add your Apple Account and create a
   **Developer ID Application** certificate.
2. Save notarization credentials once (an app-specific password from
   account.apple.com):

   ```bash
   xcrun notarytool store-credentials lectern-notary \
     --apple-id YOUR_APPLE_ID --team-id YOUR_TEAM_ID
   ```

3. Build and sign with the hardened runtime:

   ```bash
   .venv/bin/python packaging/macos/build_app.py \
     --identity "Developer ID Application: YOUR NAME (YOUR_TEAM_ID)"
   ```

4. Sign the disk image, notarize it, and staple the ticket:

   ```bash
   codesign --sign "Developer ID Application: YOUR NAME (YOUR_TEAM_ID)" --timestamp dist/Lectern-0.2.0.dmg
   xcrun notarytool submit dist/Lectern-0.2.0.dmg --keychain-profile lectern-notary --wait
   xcrun stapler staple dist/Lectern-0.2.0.dmg
   spctl --assess --type open --context context:primary-signature -v dist/Lectern-0.2.0.dmg
   ```

5. Test on a fresh user account, then attach the DMG to a GitHub Release.

`entitlements.plist` grants only `com.apple.security.device.audio-input`, the
microphone under the hardened runtime. If the frozen Python server will not
start once signed with `--identity`, check the notary log first. PyInstaller
builds sometimes also need `com.apple.security.cs.disable-library-validation`;
add only what the log or a crash asks for.

## Not done yet

- Intel Macs: the build is Apple Silicon only.
- The UI still opens in the browser; a native window would come later.
- Study notes still need Ollama installed separately. The default notes
  model (`qwen2.5:3b`) is research-licensed; swapping it needs a quality
  check on real lectures first.
- No update check yet.
