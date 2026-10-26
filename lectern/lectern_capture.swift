// lectern-capture: lossless microphone capture into fixed-length WAV segments.
//
// Replaces `ffmpeg -f avfoundation` for recording. Measured on this machine
// (macOS 27, ffmpeg 5.0): ffmpeg's avfoundation input keeps ONE pending audio
// buffer and overwrites it when the next callback arrives before ffmpeg polls
// (it sleeps 10 ms on EAGAIN). Run from a shell it kept ~88% of the audio; run
// detached from the backend it kept ~32%, so a 60 s segment held ~19 s of
// choppy speech that whisper turned into loops and word salad. The loss is
// silent: pts are wall-clock, so segments still close every 60 s.
//
// Here every CMSampleBuffer is converted and written in its own callback, and
// segments are cut by SAMPLE COUNT, so a segment always holds exactly
// --segment-seconds of audio and nothing is dropped in between.
//
// Contract with recorder.py (same as the ffmpeg invocation it replaces):
//   * writes <pattern % n> as 16 kHz mono s16le WAV with a 44-byte header;
//     sizes are 0xFFFFFFFF while open and patched on close, like ffmpeg, so
//     a crash leaves a file stdlib `wave` still reads correctly
//   * every buffer is write()n as soon as it is converted, on a dedicated I/O
//     queue: the capture callback never waits on the disk. Measured: with the
//     write inside the callback, loading the notes model on this 8 GB machine
//     stalled the disk long enough for AVFoundation to drop 1.5 s of input.
//   * prints `out_time_us=N` / `progress=continue` on stdout ONLY when audio
//     advanced, so a vanished device still shows up as a stall
//   * SIGINT/SIGTERM patch the open segment's header and exit 0
//   * ignores SIGPIPE: it must outlive the backend that reads its stdout
//   * before the Mac sleeps, and (with --pause-on-lid) when the lid closes, it
//     closes the segment, releases the mic, prints `event=sleep` or
//     `event=lid-closed`, logs "stopped (system sleep|lid closed)" and exits
//     75 -- only THEN does it let the sleep proceed. Previously the process was
//     frozen mid-recording and simply carried on with the mic live on wake.
//     SIGUSR1 takes the lid-closed path, so that path can be tested without
//     shutting the laptop.
import AVFoundation
import Foundation
import IOKit
import IOKit.pwr_mgt

let OUT_RATE = 16000.0
setvbuf(stdout, nil, _IONBF, 0)
signal(SIGPIPE, SIG_IGN)

func log(_ s: String) {            // POSIX write: a closed stderr must never crash us
    ("lectern-capture: " + s + "\n").withCString { p in _ = Darwin.write(2, p, strlen(p)) }
}

func emit(_ s: String) {           // stdout may be a dead pipe; never let that kill us
    s.withCString { p in _ = Darwin.write(1, p, strlen(p)) }
}

// ------------------------------------------------------------------ args
var deviceName: String? = nil
var pattern = ""
var segmentSeconds = 60
var startNumber = 0
var maxSeconds: Double? = nil
var pauseOnLid = false
do {
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let a = it.next() {
        switch a {
        case "--device": deviceName = it.next()
        case "--out": pattern = it.next() ?? ""
        case "--segment-seconds": segmentSeconds = Int(it.next() ?? "") ?? 60
        case "--start-number": startNumber = Int(it.next() ?? "") ?? 0
        case "--duration": maxSeconds = Double(it.next() ?? "")
        case "--pause-on-lid": pauseOnLid = true
        default: log("unknown argument \(a)"); exit(2)
        }
    }
    if pattern.isEmpty || !pattern.contains("%") {
        log("usage: lectern-capture --out /dir/seg-%06d.wav [--device NAME] "
            + "[--segment-seconds 60] [--start-number N] [--duration S] [--pause-on-lid]")
        exit(2)
    }
}

// ---------------------------------------------------------------- writer
final class SegmentWriter {
    let pattern: String
    let framesPerSegment: Int64
    var index: Int
    var fd: Int32 = -1
    var segFrames: Int64 = 0
    var totalFrames: Int64 = 0

    init(pattern: String, segmentSeconds: Int, startNumber: Int) {
        self.pattern = pattern
        self.framesPerSegment = Int64(segmentSeconds) * Int64(OUT_RATE)
        self.index = startNumber
    }

    private func le32(_ v: UInt32) -> [UInt8] { withUnsafeBytes(of: v.littleEndian, Array.init) }
    private func le16(_ v: UInt16) -> [UInt8] { withUnsafeBytes(of: v.littleEndian, Array.init) }

    private func openNext() {
        let path = String(format: pattern, index)
        fd = Darwin.open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
        if fd < 0 {
            log("cannot open \(path): \(String(cString: strerror(errno)))")
            return
        }
        var h: [UInt8] = Array("RIFF".utf8) + le32(0xFFFF_FFFF) + Array("WAVE".utf8)
        h += Array("fmt ".utf8) + le32(16) + le16(1) + le16(1)
        h += le32(UInt32(OUT_RATE)) + le32(UInt32(OUT_RATE) * 2) + le16(2) + le16(16)
        h += Array("data".utf8) + le32(0xFFFF_FFFF)
        _ = h.withUnsafeBytes { Darwin.write(fd, $0.baseAddress, 44) }
        segFrames = 0
    }

    func closeSegment() {
        guard fd >= 0 else { return }
        let dataBytes = UInt32(segFrames * 2)
        let riff = le32(36 + dataBytes), data = le32(dataBytes)
        _ = riff.withUnsafeBytes { pwrite(fd, $0.baseAddress, 4, 4) }
        _ = data.withUnsafeBytes { pwrite(fd, $0.baseAddress, 4, 40) }
        Darwin.close(fd)
        fd = -1
        index += 1
    }

    func write(_ samples: [Int16]) {
        samples.withUnsafeBufferPointer { write($0) }
    }

    func write(_ samples: UnsafeBufferPointer<Int16>) {
        var off = 0
        while off < samples.count {
            if fd < 0 { openNext(); if fd < 0 { return } }
            let room = Int(framesPerSegment - segFrames)
            let n = min(room, samples.count - off)
            let base = UnsafeRawPointer(samples.baseAddress! + off)
            var done = 0
            while done < n * 2 {
                let w = Darwin.write(fd, base + done, n * 2 - done)
                if w < 0 {
                    if errno == EINTR { continue }
                    log("write failed: \(String(cString: strerror(errno)))")
                    return
                }
                done += w
            }
            segFrames += Int64(n)
            totalFrames += Int64(n)
            off += n
            if segFrames >= framesPerSegment { closeSegment() }
        }
    }
}

// --------------------------------------------------------------- capture
final class Capture: NSObject, AVCaptureAudioDataOutputSampleBufferDelegate {
    let writer: SegmentWriter
    let io: DispatchQueue          // the ONLY queue that touches `writer`
    var framesIn: Int64 = 0
    let outFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: OUT_RATE,
                                  channels: 1, interleaved: true)!
    var converter: AVAudioConverter?
    var converterKey = ""
    var nextPts: Double? = nil
    var gaps = 0
    var gapSeconds = 0.0
    var loggedFormat = ""

    init(writer: SegmentWriter, io: DispatchQueue) {
        self.writer = writer
        self.io = io
    }

    private func hand(_ p: UnsafePointer<Int16>, _ count: Int) {
        let copy = Array(UnsafeBufferPointer(start: p, count: count))
        framesIn += Int64(count)
        io.async { self.writer.write(copy) }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let fd = CMSampleBufferGetFormatDescription(sb),
              let asbdP = CMAudioFormatDescriptionGetStreamBasicDescription(fd) else { return }
        let asbd = asbdP.pointee
        let n = CMSampleBufferGetNumSamples(sb)
        if n <= 0 { return }

        let key = "\(asbd.mSampleRate)/\(asbd.mChannelsPerFrame)/\(asbd.mBitsPerChannel)/\(asbd.mFormatFlags)"
        if key != loggedFormat {
            log("input format \(Int(asbd.mSampleRate)) Hz, \(asbd.mChannelsPerFrame) ch, "
                + "\(asbd.mBitsPerChannel) bit, flags \(asbd.mFormatFlags)")
            loggedFormat = key
        }

        // Discontinuities the OS reports are logged, never hidden.
        let pts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sb))
        if let expect = nextPts, pts.isFinite, pts - expect > 0.2 {
            gaps += 1
            gapSeconds += pts - expect
            log(String(format: "input gap of %.0f ms at %.1f s", (pts - expect) * 1000,
                       Double(framesIn) / OUT_RATE))
        }
        if pts.isFinite { nextPts = pts + Double(n) / asbd.mSampleRate }

        let isFloat = asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0
        let isBig = asbd.mFormatFlags & kAudioFormatFlagIsBigEndian != 0
        let isNonInterleaved = asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0
        if asbd.mSampleRate == OUT_RATE && asbd.mChannelsPerFrame == 1 && asbd.mBitsPerChannel == 16
            && !isFloat && !isBig && !isNonInterleaved {
            copyThrough(sb, frames: n)
        } else {
            convert(sb, fd: fd, asbd: asbd, frames: n, key: key)
        }
    }

    private func copyThrough(_ sb: CMSampleBuffer, frames n: Int) {
        var abl = AudioBufferList()
        var block: CMBlockBuffer?
        let st = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sb, bufferListSizeNeededOut: nil, bufferListOut: &abl,
            bufferListSize: MemoryLayout<AudioBufferList>.size, blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil, flags: 0, blockBufferOut: &block)
        guard st == noErr, let p = abl.mBuffers.mData else {
            log("could not read sample buffer (\(st))")
            return
        }
        hand(p.assumingMemoryBound(to: Int16.self), Int(abl.mBuffers.mDataByteSize) / 2)
    }

    private func convert(_ sb: CMSampleBuffer, fd: CMFormatDescription,
                         asbd: AudioStreamBasicDescription, frames n: Int, key: String) {
        let inFormat = AVAudioFormat(cmAudioFormatDescription: fd)
        if converter == nil || key != converterKey {
            converter = AVAudioConverter(from: inFormat, to: outFormat)
            converter?.downmix = true
            converterKey = key
        }
        guard let conv = converter,
              let inBuf = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(n))
        else { log("no converter for this input format"); return }
        let st = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sb, at: 0, frameCount: Int32(n), into: inBuf.mutableAudioBufferList)
        guard st == noErr else { log("could not copy PCM (\(st))"); return }
        inBuf.frameLength = AVAudioFrameCount(n)

        let cap = AVAudioFrameCount(Double(n) * OUT_RATE / asbd.mSampleRate) + 256
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: cap) else { return }
        var fed = false
        var err: NSError?
        conv.convert(to: outBuf, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true
            status.pointee = .haveData
            return inBuf
        }
        if let err { log("convert failed: \(err.localizedDescription)"); return }
        guard let ch = outBuf.int16ChannelData else { return }
        hand(ch[0], Int(outBuf.frameLength))
    }
}

// ----------------------------------------------------------- permission
switch AVCaptureDevice.authorizationStatus(for: .audio) {
case .authorized: break
case .notDetermined:
    let sem = DispatchSemaphore(value: 0)
    var ok = false
    AVCaptureDevice.requestAccess(for: .audio) { ok = $0; sem.signal() }
    sem.wait()
    if !ok { log("Operation not permitted: microphone access was denied"); exit(3) }
default:
    log("Operation not permitted: microphone access is denied for this app "
        + "(System Settings > Privacy & Security > Microphone)")
    exit(3)
}

// --------------------------------------------------------------- device
func findDevice(_ name: String?) -> AVCaptureDevice? {
    let types: [AVCaptureDevice.DeviceType]
    if #available(macOS 14.0, *) {
        types = [.microphone, .external]
    } else {
        types = [.builtInMicrophone, .externalUnknown]
    }
    let all = AVCaptureDevice.DiscoverySession(deviceTypes: types, mediaType: .audio,
                                               position: .unspecified).devices
    if let name {
        if let d = all.first(where: { $0.localizedName == name }) { return d }
        if let d = all.first(where: { $0.localizedName.lowercased().contains(name.lowercased()) }) {
            return d
        }
        log("device \"\(name)\" not found; using the system default input")
    }
    return AVCaptureDevice.default(for: .audio)
}

guard let device = findDevice(deviceName) else {
    log("Cannot open audio device: no microphone found")
    exit(4)
}

let session = AVCaptureSession()
let writer = SegmentWriter(pattern: pattern, segmentSeconds: segmentSeconds, startNumber: startNumber)
let queue = DispatchQueue(label: "lectern.capture", qos: .userInteractive)
let ioQueue = DispatchQueue(label: "lectern.capture.io", qos: .userInitiated)
let capture = Capture(writer: writer, io: ioQueue)

do {
    let input = try AVCaptureDeviceInput(device: device)
    guard session.canAddInput(input) else { log("Cannot open audio device: input rejected"); exit(4) }
    session.addInput(input)
} catch {
    log("Cannot open audio device: \(error.localizedDescription)")
    exit(4)
}
let output = AVCaptureAudioDataOutput()
// Ask AVFoundation for whisper's native format up front. The converter in
// Capture still handles anything else it delivers, including a mid-stream
// format change (this mic has opened as both 44.1 kHz mono and 48 kHz stereo).
output.audioSettings = [
    AVFormatIDKey: kAudioFormatLinearPCM,
    AVSampleRateKey: OUT_RATE,
    AVNumberOfChannelsKey: 1,
    AVLinearPCMBitDepthKey: 16,
    AVLinearPCMIsFloatKey: false,
    AVLinearPCMIsBigEndianKey: false,
    AVLinearPCMIsNonInterleaved: false,
]
output.setSampleBufferDelegate(capture, queue: queue)
guard session.canAddOutput(output) else { log("Cannot open audio device: output rejected"); exit(4) }
session.addOutput(output)

NotificationCenter.default.addObserver(forName: AVCaptureSession.runtimeErrorNotification,
                                       object: session, queue: nil) { note in
    let e = note.userInfo?[AVCaptureSessionErrorKey] as? NSError
    log("capture runtime error: \(e?.localizedDescription ?? "unknown")")
}
NotificationCenter.default.addObserver(forName: AVCaptureDevice.wasDisconnectedNotification,
                                       object: device, queue: nil) { _ in
    log("Input/output error: \(device.localizedName) was disconnected")
}

// ------------------------------------------------------------- shutdown
let SYSTEM_PAUSE_EXIT: Int32 = 75     // recorder.py: paused by lid/sleep, not a crash
let finishLock = NSLock()
var finishing = false
var finished = false                  // segment closed, mic released, report sent
var finishCode: Int32 = 0
var deferredAcks: [() -> Void] = []   // sleep acks that arrived mid-finish

/// Close the open segment, release the mic, report, exit. Runs at most once.
///
/// A real lid close sends clamshell-closed and then will-sleep within
/// milliseconds, so the will-sleep ack can arrive while the first finish is
/// still inside stopRunning(). It is never acknowledged early: it is queued
/// and run only once the segment is closed and the mic released. After a lid
/// pause the process lingers ~2 s (mic already off) so the will-sleep that
/// follows is acknowledged by us instead of leaving the kernel waiting on a
/// client that died with the message unread.
func finish(_ why: String, code: Int32 = 0, event: String? = nil,
            beforeExit: (() -> Void)? = nil) {
    finishLock.lock()
    if finishing {
        if finished {
            finishLock.unlock()
            beforeExit?()                              // cleanup is done: ack now
            if beforeExit == nil { exit(finishCode) }  // a signal after a lid pause
            return
        }
        if let ack = beforeExit { deferredAcks.append(ack) }
        finishLock.unlock()
        return
    }
    finishing = true
    finishCode = code
    finishLock.unlock()

    session.stopRunning()                       // mic released; callbacks drain
    queue.sync {}                               // last callback has queued its write
    ioQueue.sync {
        writer.closeSegment()
        let secs = Double(writer.totalFrames) / OUT_RATE
        log(String(format: "stopped (%@): %.2f s written, %d input gap(s) totalling %.2f s",
                   why, secs, capture.gaps, capture.gapSeconds))
        var out = "out_time_us=\(writer.totalFrames * 1_000_000 / Int64(OUT_RATE))\n"
        if let event { out += "event=\(event)\n" }
        emit(out + "progress=end\n")
    }

    finishLock.lock()
    finished = true
    let late = deferredAcks
    deferredAcks = []
    finishLock.unlock()
    beforeExit?()
    for ack in late { ack() }
    if code == SYSTEM_PAUSE_EXIT && beforeExit == nil {
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { exit(code) }
        return
    }
    exit(code)
}

var signalSources: [DispatchSourceSignal] = []
for sig in [SIGINT, SIGTERM, SIGHUP] {
    signal(sig, SIG_IGN)
    let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    src.setEventHandler { finish("signal \(sig)") }
    src.resume()
    signalSources.append(src)
}
signal(SIGUSR1, SIG_IGN)
let usr1 = DispatchSource.makeSignalSource(signal: SIGUSR1, queue: .main)
usr1.setEventHandler { finish("lid closed", code: SYSTEM_PAUSE_EXIT, event: "lid-closed") }
usr1.resume()

// ------------------------------------------------------- sleep and lid
// IOKit message constants are function-like C macros, which Swift does not
// import: iokit_common_msg(x) = 0xE0000000 | x, and the power-management
// family message 0x100 is 0xE0000000 | (13 << 14) | 0x100.
let kMsgCanSystemSleep: UInt32 = 0xE000_0270
let kMsgSystemWillSleep: UInt32 = 0xE000_0280
let kMsgClamshellStateChange: UInt32 = 0xE003_4100
let kClamshellClosedBit: UInt = 1 << 0

var rootPort: io_connect_t = 0
var powerPort: IONotificationPortRef? = nil
var powerNotifier: io_object_t = 0
var clamshellNotifier: io_object_t = 0
let powerQueue = DispatchQueue(label: "lectern.capture.power", qos: .userInteractive)

rootPort = IORegisterForSystemPower(nil, &powerPort, { _, _, messageType, argument in
    switch messageType {
    case kMsgCanSystemSleep:
        IOAllowPowerChange(rootPort, Int(bitPattern: argument))     // never veto idle sleep
    case kMsgSystemWillSleep:
        let token = Int(bitPattern: argument)
        finish("system sleep", code: SYSTEM_PAUSE_EXIT, event: "sleep") {
            IOAllowPowerChange(rootPort, token)     // ack only once the mic is off
        }
    default:
        break
    }
}, &powerNotifier)

if rootPort != 0, let port = powerPort {
    IONotificationPortSetDispatchQueue(port, powerQueue)
    if pauseOnLid {
        let root = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("IOPMrootDomain"))
        if root != 0 {
            let kr = IOServiceAddInterestNotification(port, root, kIOGeneralInterest, { _, _, messageType, argument in
                if messageType == kMsgClamshellStateChange
                    && UInt(bitPattern: argument) & kClamshellClosedBit != 0 {
                    // Synchronously, on the serial power queue: the will-sleep
                    // that follows is queued behind this and can never be
                    // acknowledged before the mic is off.
                    finish("lid closed", code: SYSTEM_PAUSE_EXIT, event: "lid-closed")
                }
            }, nil, &clamshellNotifier)
            if kr != KERN_SUCCESS { log("lid notifications unavailable (\(kr))") }
            IOObjectRelease(root)
        }
    }
} else {
    log("sleep notifications unavailable; the backend will pause after wake instead")
}

// Progress, only when audio actually advanced (the backend's stall signal).
var lastReported: Int64 = -1
let ticker = DispatchSource.makeTimerSource(queue: ioQueue)
ticker.schedule(deadline: .now() + 0.5, repeating: 0.5)
ticker.setEventHandler {
    let f = writer.totalFrames
    if f != lastReported {
        lastReported = f
        emit("out_time_us=\(f * 1_000_000 / Int64(OUT_RATE))\nprogress=continue\n")
    }
    if let m = maxSeconds, Double(f) / OUT_RATE >= m {
        DispatchQueue.main.async { finish("duration reached") }
    }
}
ticker.resume()

log("recording from \"\(device.localizedName)\" into \(pattern)"
    + (pauseOnLid ? " (pauses if the lid closes)" : ""))
session.startRunning()
dispatchMain()
