// Lectern menu bar helper (macOS).
//
// A small status-bar app that lets you start and stop a recording with one
// global shortcut (default ⌘L) without opening Lectern. It only talks to the
// local Lectern server over HTTP on 127.0.0.1, and starts that server if it is
// not running. Built and installed by `python -m lectern.menubar install`.
//
//   * global hotkey via Carbon RegisterEventHotKey: no Accessibility permission
//   * the shortcut is read from GET /api/shortcut and re-registered on change
//   * start = POST /api/record/toggle (auto-titled, no dialog), same key stops
//
// Usage: LecternMenu --repo <path to the Lectern folder> [--port 8477]

import AppKit
import Carbon.HIToolbox

// ----------------------------------------------------------------- arguments
var repo = ""
var port = 8477
do {
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let a = it.next() {
        switch a {
        case "--repo": repo = it.next() ?? ""
        case "--port": port = Int(it.next() ?? "") ?? 8477
        default: break
        }
    }
}
let base = URL(string: "http://127.0.0.1:\(port)")!

// ---------------------------------------------------------------------- http
func request(_ path: String, method: String = "GET", body: [String: Any]? = nil,
             timeout: TimeInterval = 10,
             done: @escaping ([String: Any]?, Int) -> Void) {
    var req = URLRequest(url: base.appendingPathComponent(path))
    req.httpMethod = method
    req.timeoutInterval = timeout
    if let body {
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)
    }
    URLSession.shared.dataTask(with: req) { data, resp, _ in
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        let json = data.flatMap { try? JSONSerialization.jsonObject(with: $0) } as? [String: Any]
        DispatchQueue.main.async { done(json, code) }
    }.resume()
}

/// Start the Lectern server if nothing answers, then call `then(ok)`.
func ensureServer(_ then: @escaping (Bool) -> Void) {
    request("api/status", timeout: 1.5) { _, code in
        if code == 200 { return then(true) }
        guard !repo.isEmpty else { return then(false) }
        let py = URL(fileURLWithPath: repo).appendingPathComponent(".venv/bin/python")
        let logs = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Lectern/logs")
        try? FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        let logURL = logs.appendingPathComponent("server.out")
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }
        let p = Process()
        p.executableURL = py
        p.arguments = ["-m", "lectern"]
        p.currentDirectoryURL = URL(fileURLWithPath: repo)
        if let log = try? FileHandle(forWritingTo: logURL) {
            log.seekToEndOfFile()
            p.standardOutput = log
            p.standardError = log
        }
        do { try p.run() } catch { return then(false) }
        // Wait for it to answer (first start builds the mic helper: be patient).
        var tries = 0
        func check() {
            request("api/status", timeout: 1) { _, code in
                if code == 200 { return then(true) }
                tries += 1
                if tries > 60 { return then(false) }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5, execute: check)
            }
        }
        check()
    }
}

// -------------------------------------------------------------------- hotkey
let keyCodes: [Character: Int] = [
    "A": kVK_ANSI_A, "B": kVK_ANSI_B, "C": kVK_ANSI_C, "D": kVK_ANSI_D, "E": kVK_ANSI_E,
    "F": kVK_ANSI_F, "G": kVK_ANSI_G, "H": kVK_ANSI_H, "I": kVK_ANSI_I, "J": kVK_ANSI_J,
    "K": kVK_ANSI_K, "L": kVK_ANSI_L, "M": kVK_ANSI_M, "N": kVK_ANSI_N, "O": kVK_ANSI_O,
    "P": kVK_ANSI_P, "Q": kVK_ANSI_Q, "R": kVK_ANSI_R, "S": kVK_ANSI_S, "T": kVK_ANSI_T,
    "U": kVK_ANSI_U, "V": kVK_ANSI_V, "W": kVK_ANSI_W, "X": kVK_ANSI_X, "Y": kVK_ANSI_Y,
    "Z": kVK_ANSI_Z, "0": kVK_ANSI_0, "1": kVK_ANSI_1, "2": kVK_ANSI_2, "3": kVK_ANSI_3,
    "4": kVK_ANSI_4, "5": kVK_ANSI_5, "6": kVK_ANSI_6, "7": kVK_ANSI_7, "8": kVK_ANSI_8,
    "9": kVK_ANSI_9,
]
let modBits: [String: Int] = ["cmd": cmdKey, "ctrl": controlKey, "opt": optionKey, "shift": shiftKey]
let modGlyphs: [(String, String)] = [("ctrl", "⌃"), ("opt", "⌥"), ("shift", "⇧"), ("cmd", "⌘")]

struct Shortcut: Equatable {
    var enabled = true
    var key = "L"
    var mods = ["cmd"]
    var label: String {
        modGlyphs.filter { mods.contains($0.0) }.map { $0.1 }.joined() + key
    }
}

var hotKeyRef: EventHotKeyRef?
var current = Shortcut(enabled: false, key: "", mods: [])
var onHotKey: () -> Void = {}

func installHotKeyHandler() {
    var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard),
                             eventKind: UInt32(kEventHotKeyPressed))
    InstallEventHandler(GetApplicationEventTarget(), { _, _, _ in
        DispatchQueue.main.async { onHotKey() }
        return noErr
    }, 1, &spec, nil, nil)
}

/// Returns false if macOS refused the combination (another app owns it).
@discardableResult
func register(_ sc: Shortcut) -> Bool {
    if let ref = hotKeyRef { UnregisterEventHotKey(ref); hotKeyRef = nil }
    current = sc
    guard sc.enabled, let ch = sc.key.uppercased().first, let code = keyCodes[ch] else { return true }
    let mods = sc.mods.reduce(0) { $0 | (modBits[$1] ?? 0) }
    let id = EventHotKeyID(signature: OSType(0x4C43_544E), id: 1)   // 'LCTN'
    let err = RegisterEventHotKey(UInt32(code), UInt32(mods), id,
                                  GetApplicationEventTarget(), 0, &hotKeyRef)
    return err == noErr
}

// ------------------------------------------------------------------- the app
final class Controller: NSObject, NSMenuDelegate {
    let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    var status: [String: Any] = [:]
    var online = false
    var busy = false
    var hotkeyFailed = false

    override init() {
        super.init()
        let menu = NSMenu()
        menu.delegate = self
        item.menu = menu
        render()
        onHotKey = { [weak self] in self?.toggle() }
        installHotKeyHandler()
        ensureServer { [weak self] ok in
            self?.online = ok
            self?.refreshShortcut()
            self?.poll()
        }
        Timer.scheduledTimer(withTimeInterval: 15, repeats: true) { [weak self] _ in
            self?.refreshShortcut()
        }
    }

    var recording: Bool { status["recording"] as? Bool ?? false }

    func refreshShortcut() {
        request("api/shortcut", timeout: 3) { json, code in
            guard code == 200, let json else { return }
            let sc = Shortcut(enabled: json["enabled"] as? Bool ?? true,
                              key: json["key"] as? String ?? "L",
                              mods: json["mods"] as? [String] ?? ["cmd"])
            if sc != current { self.hotkeyFailed = !register(sc) }
        }
    }

    func poll() {
        request("api/status", timeout: 3) { [weak self] json, code in
            guard let self else { return }
            self.online = code == 200
            if let json { self.status = json }
            self.render()
            DispatchQueue.main.asyncAfter(deadline: .now() + (self.recording ? 1 : 3)) { self.poll() }
        }
    }

    func render() {
        guard let button = item.button else { return }
        let symbol = recording ? "record.circle.fill" : "mic"
        let img = NSImage(systemSymbolName: symbol, accessibilityDescription: "Lectern")
        img?.isTemplate = !recording
        button.image = img
        button.contentTintColor = recording ? .systemRed : nil
        if recording, let s = status["elapsed_s"] as? Double {
            let t = Int(s)
            button.title = t >= 3600 ? String(format: " %d:%02d:%02d", t / 3600, t / 60 % 60, t % 60)
                                     : String(format: " %d:%02d", t / 60, t % 60)
        } else {
            button.title = ""
        }
        button.imagePosition = .imageLeading
    }

    // --- actions
    func toggle() {
        guard !busy else { return }
        busy = true
        let wasRecording = recording
        ensureServer { ok in
            guard ok else { self.busy = false; return self.alert("Lectern isn't running and could not be started.") }
            request("api/record/toggle", method: "POST", timeout: 30) { json, code in
                self.busy = false
                if code == 200 {
                    NSSound(named: wasRecording || json?["action"] as? String == "stopped" ? "Pop" : "Tink")?.play()
                } else {
                    NSSound.beep()
                    self.alert(json?["error"] as? String ?? "Could not start or stop the recording.",
                               detail: json?["detail"] as? String)
                }
                self.poll()
            }
        }
    }

    @objc func toggleAction() { toggle() }

    @objc func pause() {
        request("api/record/pause", method: "POST") { json, code in
            if code != 200 { self.alert(json?["error"] as? String ?? "Could not pause.") }
            self.poll()
        }
    }

    @objc func resume() {
        guard let ls = status["last_stop"] as? [String: Any], let id = ls["lecture_id"] as? String else { return }
        request("api/record/resume", method: "POST", body: ["lecture_id": id], timeout: 30) { json, code in
            if code != 200 { self.alert(json?["error"] as? String ?? "Could not resume.") }
            self.poll()
        }
    }

    @objc func openApp() {
        ensureServer { _ in NSWorkspace.shared.open(base) }
    }

    @objc func openShortcutSettings() {
        ensureServer { _ in NSWorkspace.shared.open(URL(string: "\(base.absoluteString)/#settings")!) }
    }

    @objc func quit() { NSApp.terminate(nil) }

    func alert(_ text: String, detail: String? = nil) {
        NSApp.activate(ignoringOtherApps: true)
        let a = NSAlert()
        a.messageText = text
        if let detail, !detail.isEmpty { a.informativeText = detail }
        a.runModal()
    }

    // --- menu, rebuilt each time it opens so it is never stale
    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()
        func add(_ title: String, _ action: Selector?, enabled: Bool = true) -> NSMenuItem {
            let mi = NSMenuItem(title: title, action: action, keyEquivalent: "")
            mi.target = self
            mi.isEnabled = enabled && action != nil
            menu.addItem(mi)
            return mi
        }
        let keyHint = current.enabled ? "   \(current.label)" : ""
        if !online {
            _ = add("Lectern is not running", nil)
            _ = add("Start Lectern", #selector(openApp))
        } else if recording {
            _ = add("Recording: \(status["title"] as? String ?? "")", nil)
            _ = add("Stop & transcribe\(keyHint)", #selector(toggleAction))
            _ = add("Pause", #selector(pause))
        } else {
            _ = add("Start recording\(keyHint)", #selector(toggleAction))
            if let ls = status["last_stop"] as? [String: Any], ls["paused"] as? Bool == true {
                _ = add("Continue paused lecture", #selector(resume))
            }
        }
        if hotkeyFailed {
            _ = add("Shortcut \(current.label) is taken by another app", nil)
        }
        menu.addItem(.separator())
        _ = add("Open Lectern", #selector(openApp))
        _ = add("Change shortcut…", #selector(openShortcutSettings))
        menu.addItem(.separator())
        _ = add("Quit menu bar helper", #selector(quit))
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)           // menu bar only, no Dock icon
let controller = Controller()
app.run()
