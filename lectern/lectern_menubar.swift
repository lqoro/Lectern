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
//
// Inside Lectern.app (packaging/macos/build_app.py) it needs no --repo: it runs
// the bundled server in Contents/Resources/backend, points it at the bundled
// tools in Contents/Resources/bin, and shows Lectern in its own window (a
// WKWebView on the local server) with a Dock icon and the usual Mac menus.

import AppKit
import Carbon.HIToolbox
import UniformTypeIdentifiers
import WebKit

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

// Lectern.app ships the server and its tools inside the bundle.
let resources = Bundle.main.resourceURL ?? URL(fileURLWithPath: "/nonexistent")
let bundledServer = resources.appendingPathComponent("backend/lectern-server/lectern-server")
let bundledBin = resources.appendingPathComponent("bin")
// No --repo means this is the app. Decided by how we were started, not by
// whether the server file is there: a packaging mistake (an Xcode copy phase
// that drops the folder or its executable bit) must show an error, not quietly
// turn the app into the menu-bar-only developer helper.
let isBundled = repo.isEmpty
let bundledServerMissing = isBundled && !FileManager.default.isExecutableFile(atPath: bundledServer.path)
// The server this app started, stopped again when the app quits. Left running,
// the next launch (after an update, or from Applications instead of the disk
// image) would silently keep talking to the old copy.
var serverProcess: Process?

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
/// Stop a leftover Lectern server (an older install still running on our port)
/// so the app never shows its old interface. Never touches a recording in progress.
func stopStaleServer() {
    let find = Process()
    find.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
    find.arguments = ["-ti", "tcp:\(port)", "-sTCP:LISTEN"]
    let pipe = Pipe()
    find.standardOutput = pipe
    find.standardError = FileHandle.nullDevice
    guard (try? find.run()) != nil else { return }
    find.waitUntilExit()
    let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    for pid in out.split(whereSeparator: \.isNewline).compactMap({ Int32($0) }) { kill(pid, SIGTERM) }
    Thread.sleep(forTimeInterval: 1.0)
}

func ensureServer(_ then: @escaping (Bool) -> Void) {
    request("api/status", timeout: 1.5) { json, code in
        if code == 200 {
            let mine = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String
            let stale = isBundled && json?["version"] as? String != mine
            if !stale || json?["recording"] as? Bool == true { return then(true) }
            stopStaleServer()
        }
        guard !bundledServerMissing else { return then(false) }
        let logs = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Lectern/logs")
        try? FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        let logURL = logs.appendingPathComponent("server.out")
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }
        let p = Process()
        if isBundled {
            p.executableURL = bundledServer
            p.currentDirectoryURL = FileManager.default.homeDirectoryForCurrentUser
            var env = ProcessInfo.processInfo.environment
            env["LECTERN_BIN_DIR"] = bundledBin.path
            env["LECTERN_PORT"] = String(port)
            p.environment = env
        } else {
            p.executableURL = URL(fileURLWithPath: repo).appendingPathComponent(".venv/bin/python")
            p.arguments = ["-m", "lectern"]
            p.currentDirectoryURL = URL(fileURLWithPath: repo)
        }
        if let log = try? FileHandle(forWritingTo: logURL) {
            log.seekToEndOfFile()
            p.standardOutput = log
            p.standardError = log
        }
        do { try p.run() } catch { return then(false) }
        serverProcess = p
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


// -------------------------------------------------------------- the window
/// Lectern's own window in the bundled app: the web UI on 127.0.0.1 in a
/// WKWebView. Links that leave the local server (help, donate, licences) and
/// file downloads open in the default browser.
final class MainWindow: NSObject, WKNavigationDelegate, WKUIDelegate, NSWindowDelegate {
    let window: NSWindow
    let web: WKWebView

    override init() {
        let cfg = WKWebViewConfiguration()
        // The web UI shows Mac-app-only parts (the donate button) when it sees this.
        cfg.applicationNameForUserAgent = "LecternMac/1"
        web = WKWebView(frame: .zero, configuration: cfg)
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1240, height: 820),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        super.init()
        web.navigationDelegate = self
        web.uiDelegate = self
        window.title = "Lectern"
        window.minSize = NSSize(width: 380, height: 520)
        window.contentView = web
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.setFrameAutosaveName("LecternMain")
        if window.frame.origin == .zero { window.center() }
        web.load(URLRequest(url: base))
    }

    func show(hash: String? = nil) {
        if let hash {
            web.evaluateJavaScript("location.hash = \(String(reflecting: hash)); if (\(String(reflecting: hash)) === '#settings') document.getElementById('settingsBtn').click();")
        }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func isLocal(_ url: URL?) -> Bool {
        guard let url else { return true }
        return url.scheme == "about" || (url.host == "127.0.0.1" && url.port == port)
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if !isLocal(action.request.url), let url = action.request.url {
            NSWorkspace.shared.open(url)
            return decisionHandler(.cancel)
        }
        decisionHandler(.allow)
    }

    // A download (Content-Disposition: attachment) goes to the browser, which saves it.
    func webView(_ webView: WKWebView, decidePolicyFor response: WKNavigationResponse,
                 decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        let http = response.response as? HTTPURLResponse
        let disposition = http?.value(forHTTPHeaderField: "Content-Disposition") ?? ""
        if disposition.lowercased().hasPrefix("attachment") || !response.canShowMIMEType,
           let url = response.response.url {
            NSWorkspace.shared.open(url)
            return decisionHandler(.cancel)
        }
        decisionHandler(.allow)
    }

    // target="_blank" links.
    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
                 for action: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let url = action.request.url { NSWorkspace.shared.open(url) }
        return nil
    }

    // <input type="file"> for Import.
    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.audio, .movie, .audiovisualContent]
        panel.beginSheetModal(for: window) { r in completionHandler(r == .OK ? panel.urls : nil) }
    }

    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) {
        let a = NSAlert()
        a.messageText = message
        a.beginSheetModal(for: window) { _ in completionHandler() }
    }

    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        let a = NSAlert()
        a.messageText = message
        a.addButton(withTitle: "OK")
        a.addButton(withTitle: "Cancel")
        a.beginSheetModal(for: window) { r in completionHandler(r == .alertFirstButtonReturn) }
    }

    func webView(_ webView: WKWebView, runJavaScriptTextInputPanelWithPrompt prompt: String,
                 defaultText: String?, initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping (String?) -> Void) {
        let a = NSAlert()
        a.messageText = prompt
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 300, height: 24))
        field.stringValue = defaultText ?? ""
        a.accessoryView = field
        a.addButton(withTitle: "OK")
        a.addButton(withTitle: "Cancel")
        a.window.initialFirstResponder = field
        a.beginSheetModal(for: window) { r in
            completionHandler(r == .alertFirstButtonReturn ? field.stringValue : nil)
        }
    }

    // If the server restarts under us, reload once it answers again.
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!,
                 withError error: Error) {
        ensureServer { ok in if ok { webView.load(URLRequest(url: base)) } }
    }
}

/// File, Edit and Window menus, so copy, paste, ⌘W and ⌘Q work as on any Mac app.
func installMainMenu(settings: Selector, target: AnyObject) {
    let main = NSMenu()
    func submenu(_ title: String) -> NSMenu {
        let item = NSMenuItem(title: title, action: nil, keyEquivalent: "")
        let m = NSMenu(title: title)
        item.submenu = m
        main.addItem(item)
        return m
    }
    let appMenu = submenu("Lectern")
    appMenu.addItem(withTitle: "About Lectern",
                    action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
    appMenu.addItem(.separator())
    let s = appMenu.addItem(withTitle: "Settings…", action: settings, keyEquivalent: ",")
    s.target = target
    appMenu.addItem(.separator())
    appMenu.addItem(withTitle: "Hide Lectern", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
    appMenu.addItem(withTitle: "Quit Lectern", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
    let edit = submenu("Edit")
    edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
    edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
    edit.addItem(.separator())
    edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
    edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
    edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
    edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
    let win = submenu("Window")
    win.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
    win.addItem(withTitle: "Close", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
    NSApp.mainMenu = main
    NSApp.windowsMenu = win
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
            // Opening Lectern.app should open Lectern, not just a menu bar icon.
            if ok && isBundled { openLecternUI() }
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
        ensureServer { _ in openLecternUI() }
    }

    @objc func openShortcutSettings() {
        ensureServer { _ in openLecternUI(hash: "#settings") }
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
        _ = add(isBundled ? "Quit Lectern" : "Quit menu bar helper", #selector(quit))
    }
}

/// The bundled app shows its own window; the developer helper (`--repo`) keeps
/// opening the browser, as before.
var mainWindow: MainWindow?
func openLecternUI(hash: String? = nil) {
    guard isBundled else {
        NSWorkspace.shared.open(URL(string: base.absoluteString + "/" + (hash ?? ""))!)
        return
    }
    if mainWindow == nil { mainWindow = MainWindow() }
    mainWindow?.show(hash: hash)
}

/// Clicking the Dock icon (or opening Lectern.app again) brings the window back.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        guard bundledServerMissing else { return }
        let a = NSAlert()
        a.messageText = "Lectern is damaged"
        a.informativeText = "Its server is missing from \(bundledServer.path). Download Lectern again."
        a.runModal()
        NSApp.terminate(nil)
    }

    func applicationWillTerminate(_ notification: Notification) {
        guard let p = serverProcess, p.isRunning else { return }
        p.terminate()          // SIGTERM; a recording writer is a separate process and keeps going
        let deadline = Date().addingTimeInterval(3)
        while p.isRunning && Date() < deadline { usleep(50_000) }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows: Bool) -> Bool {
        if isBundled { ensureServer { _ in openLecternUI() } }
        return false
    }
}

let app = NSApplication.shared
// Lectern.app is a normal app with a Dock icon; the developer helper stays menu-bar only.
app.setActivationPolicy(isBundled ? .regular : .accessory)
let delegate = AppDelegate()
app.delegate = delegate
let controller = Controller()
if isBundled { installMainMenu(settings: #selector(Controller.openShortcutSettings), target: controller) }
app.run()
