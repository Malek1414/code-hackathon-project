import AVFoundation
import SwiftUI

// Viewfinder chrome: broadcast-camera vocabulary — tally lamps, a graduated
// pan scale with a live needle, monospaced telemetry, AF-corner target box.
// Orange = the printed parts' PLA; blue = servo; red = REC only.

private extension Color {
    static let pla = Color(red: 0.92, green: 0.39, blue: 0.08)
    static let servo = Color(red: 0.15, green: 0.38, blue: 0.90)
    static let chrome = Color(white: 0.55)
}

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @StateObject private var pan = PanController()
    @StateObject private var heart = HeartRateMonitor()
    @StateObject private var whoop = WhoopAuthCoordinator.shared
    @StateObject private var level = MotionLevel()
    @State private var tracker = SubjectTracker()
    @State private var subjectBox: CGRect?
    @State private var players: [CGRect] = []   // detected people, Vision-normalized
    @State private var bridgeHost = "192.168.1.10"
    @State private var showSettings = false
    @State private var recordingStarted: Date?
    @State private var lostFlashUntil: Date?
    @State private var sweptRange: ClosedRange<Double> = 90...90
    // auto-reacquire: retry the last known box for a short window after loss
    @State private var reacquireBox: CGRect?
    @State private var reacquireUntil: Date = .distantPast
    @State private var frameCount = 0
    // session summary counters (active while recording)
    @State private var recFrames = 0
    @State private var recTargetFrames = 0
    @State private var recHRRange: ClosedRange<Int>?
    @State private var recSwept: ClosedRange<Double> = 90...90
    @State private var summary: SessionSummary?
    @AppStorage("didOnboard") private var didOnboard = false
    @State private var showGuide = false
    // WHOOP app credentials (reuse the HUSTLR registration: add the redirect
    // URI followcam://whoop-callback to it at developer.whoop.com)
    @State private var whoopClientID = Keychain.get(KeychainKey.whoopClientID)
        ?? "fcdd77dd-47c5-4530-857a-014a480437b0"
    @State private var whoopSecret = ""
    @State private var whoopStatus: String?

    var body: some View {
        ZStack {
            CameraPreview(layer: camera.previewLayer)
                .ignoresSafeArea()
                .onTapGesture { location in
                    let tapROI = camera.normalizedRect(aroundLayerPoint: location)
                    let tapCenter = CGPoint(x: tapROI.midX, y: tapROI.midY)
                    // snap to the detected player nearest the tap; raw ROI fallback
                    let hit = players.min(by: {
                        hypot($0.midX - tapCenter.x, $0.midY - tapCenter.y) <
                        hypot($1.midX - tapCenter.x, $1.midY - tapCenter.y)
                    })
                    if let hit, hypot(hit.midX - tapCenter.x, hit.midY - tapCenter.y) < 0.25 {
                        tracker.select(roi: hit)
                    } else {
                        tracker.select(roi: tapROI)
                    }
                    lostFlashUntil = nil
                    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                }

            PlayerRings(players: players, locked: subjectBox).allowsHitTesting(false)

            if let box = subjectBox {
                TargetBrackets(box: box).allowsHitTesting(false)
            }

            VStack(spacing: 0) {
                statusStrip
                PanScale(angle: pan.angle)
                    .frame(height: 44)
                    .padding(.horizontal, 24)
                LevelLine(roll: level.rollDegrees, isLevel: level.isLevel)
                    .frame(height: 16)
                Spacer()
                HStack(alignment: .bottom) {
                    if let until = lostFlashUntil, until > Date() {
                        Text("TARGET LOST — TAP TO REACQUIRE")
                            .font(.system(size: 12, weight: .bold, design: .monospaced))
                            .foregroundColor(.white)
                            .padding(.horizontal, 12).padding(.vertical, 6)
                            .background(Color.red)
                    }
                    Spacer()
                    CoverageWedge(angle: pan.angle, swept: sweptRange)
                        .frame(width: 92, height: 58)
                        .padding(.trailing, 20)
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 8)
                bottomBar
            }
        }
        .preferredColorScheme(.dark)
        .sheet(isPresented: $showSettings) { settingsSheet }
        .sheet(item: $summary) { s in SummaryCard(summary: s) }
        .overlay {
            if showGuide {
                SetupGuide(rigLinked: pan.connected,
                           targetSet: tracker.isTracking,
                           recording: camera.isRecording) {
                    showGuide = false
                    didOnboard = true
                }
            }
        }
        .onAppear {
            camera.onFrame = { pixelBuffer in
                let box = tracker.track(in: pixelBuffer)   // Vision work stays off-main
                let detected = tracker.frameIndex % 3 == 0 ? tracker.humans(in: pixelBuffer) : nil
                DispatchQueue.main.async {
                    if let detected { players = detected }
                    handleFrame(box: box)
                }
            }
            camera.start()
            level.start()
            if !didOnboard { showGuide = true }
        }
    }

    // MARK: per-frame state (main thread)

    private func handleFrame(box: CGRect?) {
        frameCount += 1
        if recordingStarted != nil {
            recFrames += 1
            if box != nil { recTargetFrames += 1 }
            if let bpm = heart.bpm {
                recHRRange = recHRRange.map { min($0.lowerBound, bpm)...max($0.upperBound, bpm) } ?? bpm...bpm
            }
            recSwept = min(recSwept.lowerBound, pan.angle)...max(recSwept.upperBound, pan.angle)
        }
        guard let box = box else {
            if subjectBox != nil {   // just lost the target
                lostFlashUntil = Date().addingTimeInterval(2.5)
                reacquireBox = subjectBox
                reacquireUntil = Date().addingTimeInterval(3.0)
                UINotificationFeedbackGenerator().notificationOccurred(.warning)
            }
            subjectBox = nil
            // auto-reacquire: prefer the detected player nearest the last known
            // spot; fall back to a widened blind reseed
            if let lost = reacquireBox, Date() < reacquireUntil, frameCount % 10 == 0 {
                let near = players.min(by: {
                    hypot($0.midX - lost.midX, $0.midY - lost.midY) <
                    hypot($1.midX - lost.midX, $1.midY - lost.midY)
                })
                if let near, hypot(near.midX - lost.midX, near.midY - lost.midY) < 0.3 {
                    tracker.select(roi: near)
                } else {
                    tracker.select(roi: lost.insetBy(dx: -lost.width * 0.35, dy: -lost.height * 0.35))
                }
            }
            return
        }
        if subjectBox == nil, reacquireBox != nil {   // reacquired
            reacquireBox = nil
            lostFlashUntil = nil
            UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
        }
        subjectBox = box
        pan.update(centerX: box.midX)
        sweptRange = min(sweptRange.lowerBound, pan.angle)...max(sweptRange.upperBound, pan.angle)
    }

    // MARK: top status strip — tally lamps + timecode

    private var statusStrip: some View {
        HStack(spacing: 18) {
            tally("RIG", on: pan.connected, color: .servo) {
                pan.connect(host: bridgeHost)
            }
            tally("HR", on: heart.bpm != nil, color: .red,
                  detail: heart.bpm.map { "\($0)" }) {
                heart.startScan()
            }
            if tracker.isTracking {
                Button {
                    tracker.clear()
                    subjectBox = nil
                    lostFlashUntil = nil
                    sweptRange = 90...90
                    pan.recenter()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "xmark").font(.system(size: 10, weight: .bold))
                        Text("LOCK")
                            .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    }
                    .foregroundColor(.pla)
                }
            }
            Spacer()
            if let started = recordingStarted {
                TimecodeView(since: started)
            }
            Button { showSettings = true } label: {
                Image(systemName: "slider.horizontal.3")
                    .foregroundColor(.chrome)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .background(.black.opacity(0.55))
    }

    private func tally(_ label: String, on: Bool, color: Color,
                       detail: String? = nil, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Circle()
                    .fill(on ? color : Color(white: 0.25))
                    .frame(width: 8, height: 8)
                Text(detail ?? label)
                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    .foregroundColor(on ? .white : .chrome)
            }
        }
    }

    // MARK: bottom bar — record + target state

    private var bottomBar: some View {
        ZStack {
            RecordButton(isRecording: camera.isRecording) {
                UIImpactFeedbackGenerator(style: .rigid).impactOccurred()
                if camera.isRecording {
                    camera.stopRecording()
                    heart.endLog(stamp: Int(Date().timeIntervalSince1970))
                    if let started = recordingStarted {
                        summary = SessionSummary(
                            duration: Date().timeIntervalSince(started),
                            coverageDeg: recSwept.upperBound - recSwept.lowerBound,
                            targetHeld: recFrames > 0 ? Double(recTargetFrames) / Double(recFrames) : 0,
                            hr: recHRRange)
                    }
                    recordingStarted = nil
                } else {
                    recFrames = 0
                    recTargetFrames = 0
                    recHRRange = nil
                    recSwept = pan.angle...pan.angle
                    heart.beginLog()
                    camera.startRecording()
                    recordingStarted = Date()
                }
            }
        }
        .padding(.bottom, 30)
    }

    // MARK: settings sheet

    private var settingsSheet: some View {
        NavigationStack {
            Form {
                Section("Rig link") {
                    TextField("Laptop IP", text: $bridgeHost)
                        .font(.system(.body, design: .monospaced))
                        .keyboardType(.decimalPad)
                    Toggle("Invert pan direction", isOn: $pan.invert)
                    Button(pan.connected ? "Reconnect" : "Connect") {
                        pan.connect(host: bridgeHost)
                    }
                }
                Section("WHOOP") {
                    TextField("Client ID", text: $whoopClientID)
                        .font(.system(.footnote, design: .monospaced))
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                    SecureField("Client secret", text: $whoopSecret)
                        .font(.system(.footnote, design: .monospaced))
                    Button("Save keys") {
                        WhoopClient.configure(clientID: whoopClientID, secret: whoopSecret)
                        whoopSecret = ""
                        whoopStatus = "Keys saved"
                    }
                    .disabled(whoopClientID.isEmpty || whoopSecret.isEmpty)
                    Button(whoop.connected ? "Connected — sync latest workout" : "Connect WHOOP") {
                        Task {
                            do {
                                if !whoop.connected { try await whoop.connect() }
                                let w = try await WhoopClient.shared.exportLatestWorkout()
                                whoopStatus = w.map {
                                    "Synced workout: strain \($0.score?.strain.map { String(format: "%.1f", $0) } ?? "–"), max HR \($0.score?.maxHeartRate.map { String(Int($0)) } ?? "–")"
                                } ?? "Connected — no workout in the last 24h"
                            } catch {
                                whoopStatus = error.localizedDescription
                            }
                        }
                    }
                    if let status = whoopStatus {
                        Text(status)
                            .font(.footnote)
                            .foregroundColor(.secondary)
                    }
                }
                Section {
                    Button("Show setup guide") {
                        showSettings = false
                        showGuide = true
                    }
                }
            }
            .navigationTitle("FollowCam")
            .navigationBarTitleDisplayMode(.inline)
        }
        .presentationDetents([.medium])
    }
}

// MARK: - pan scale: graduated 40-140 ruler with live needle

struct PanScale: View {
    let angle: Double

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let pad: CGFloat = 22   // inset so the 40/140 ticks, labels and the
                                    // needle stay fully visible at the extremes
            let x: (Double) -> CGFloat = { pad + (w - 2 * pad) * ($0 - 40) / 100 }
            ZStack(alignment: .topLeading) {
                // ticks every 5 deg, tall every 25
                ForEach(Array(stride(from: 40, through: 140, by: 5)), id: \.self) { deg in
                    Rectangle()
                        .fill(Color.chrome.opacity(deg % 25 == 15 ? 0.9 : 0.45))
                        .frame(width: 1, height: deg % 25 == 15 ? 12 : 7)
                        .position(x: x(Double(deg)), y: 8)
                }
                ForEach([40, 90, 140], id: \.self) { deg in
                    Text("\(deg)")
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .foregroundColor(.chrome)
                        .position(x: x(Double(deg)), y: 26)
                }
                // needle + readout share the same x — no parallax at any angle
                Rectangle()
                    .fill(Color.pla)
                    .frame(width: 2, height: 16)
                    .position(x: x(angle), y: 8)
                Text("\(Int(angle))°")
                    .font(.system(size: 13, weight: .bold, design: .monospaced))
                    .foregroundColor(.pla)
                    .position(x: x(angle), y: 38)
            }
            .animation(.linear(duration: 0.08), value: angle)
        }
        .padding(.top, 6)
        .background(.black.opacity(0.55))
    }
}

// MARK: - session summary after each recording

struct SessionSummary: Identifiable {
    let id = UUID()
    let duration: TimeInterval
    let coverageDeg: Double     // swept range while recording
    let targetHeld: Double      // 0...1 share of frames with a locked target
    let hr: ClosedRange<Int>?
}

struct SummaryCard: View {
    let summary: SessionSummary

    private func stat(_ value: String, _ label: String) -> some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.system(size: 30, weight: .bold, design: .monospaced))
                .foregroundColor(.pla)
            Text(label)
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundColor(.chrome)
        }
        .frame(maxWidth: .infinity)
    }

    var body: some View {
        VStack(spacing: 26) {
            Text("SESSION SAVED TO PHOTOS")
                .font(.system(size: 13, weight: .bold, design: .monospaced))
                .foregroundColor(.white)
            HStack {
                stat(String(format: "%d:%02d", Int(summary.duration) / 60,
                            Int(summary.duration) % 60), "RECORDED")
                stat("\(Int(summary.coverageDeg))°", "PAN COVERED")
                stat("\(Int(summary.targetHeld * 100))%", "TARGET HELD")
            }
            if let hr = summary.hr {
                Text("♥ \(hr.lowerBound)–\(hr.upperBound) bpm")
                    .font(.system(size: 16, weight: .semibold, design: .monospaced))
                    .foregroundColor(.red)
            }
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
        .presentationDetents([.height(260)])
    }
}

// MARK: - first-launch setup guide (doubles as the demo script)

struct SetupGuide: View {
    let rigLinked: Bool
    let targetSet: Bool
    let recording: Bool
    let dismiss: () -> Void

    private func step(_ n: Int, _ text: String, done: Bool) -> some View {
        HStack(spacing: 14) {
            Image(systemName: done ? "checkmark.circle.fill" : "\(n).circle")
                .font(.system(size: 26))
                .foregroundColor(done ? .green : .chrome)
            Text(text)
                .font(.system(size: 15, weight: .semibold, design: .monospaced))
                .foregroundColor(done ? .white : .chrome)
            Spacer()
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            Text("FOLLOWCAM SETUP")
                .font(.system(size: 14, weight: .bold, design: .monospaced))
                .foregroundColor(.pla)
            step(1, "Link the rig — settings icon, enter laptop IP", done: rigLinked)
            step(2, "Tap the ball (or its carrier) on screen", done: targetSet)
            step(3, "Hit record — the rig does the rest", done: recording)
            Button(action: dismiss) {
                Text(rigLinked && targetSet && recording ? "ROLLING — DISMISS" : "GOT IT")
                    .font(.system(size: 14, weight: .bold, design: .monospaced))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(Color.pla)
                    .foregroundColor(.black)
            }
        }
        .padding(24)
        .background(Color.black.opacity(0.92))
        .overlay(Rectangle().stroke(Color.pla, lineWidth: 1))
        .padding(32)
    }
}

// MARK: - bubble level under the pan scale (green = tripod level)

struct LevelLine: View {
    let roll: Double
    let isLevel: Bool

    var body: some View {
        ZStack {
            Rectangle().fill(Color(white: 0.25)).frame(width: 72, height: 1)
            Rectangle()
                .fill(isLevel ? Color.green : Color.chrome)
                .frame(width: 60, height: 2)
                .rotationEffect(.degrees(-roll))
                .animation(.linear(duration: 0.1), value: roll)
        }
        .frame(maxWidth: .infinity)
        .background(.black.opacity(0.55))
    }
}

// MARK: - court-coverage wedge: what the sweep has covered this play

struct CoverageWedge: View {
    let angle: Double       // current servo angle
    let swept: ClosedRange<Double>

    private func rad(_ servoDeg: Double) -> Double {
        // servo 40..140 -> screen angle: 90 points up, ends spread ±50
        (270 + (servoDeg - 90)) * .pi / 180
    }

    var body: some View {
        Canvas { ctx, size in
            let c = CGPoint(x: size.width / 2, y: size.height - 6)
            let r = size.height - 12
            var lim = Path()
            lim.addArc(center: c, radius: r, startAngle: .radians(rad(40)),
                       endAngle: .radians(rad(140)), clockwise: false)
            ctx.stroke(lim, with: .color(.init(white: 0.35)), lineWidth: 1)
            if swept.upperBound - swept.lowerBound > 0.5 {
                var fill = Path()
                fill.move(to: c)
                fill.addArc(center: c, radius: r, startAngle: .radians(rad(swept.lowerBound)),
                            endAngle: .radians(rad(swept.upperBound)), clockwise: false)
                fill.closeSubpath()
                ctx.fill(fill, with: .color(.servo.opacity(0.28)))
            }
            var needle = Path()
            needle.move(to: c)
            needle.addLine(to: CGPoint(x: c.x + r * cos(rad(angle)),
                                       y: c.y + r * sin(rad(angle))))
            ctx.stroke(needle, with: .color(.pla), lineWidth: 2)
            ctx.fill(Path(ellipseIn: CGRect(x: c.x - 3, y: c.y - 3, width: 6, height: 6)),
                     with: .color(.pla))
        }
        .background(.black.opacity(0.55))
    }
}

// MARK: - detection rings around every player (tap one to lock it)

struct PlayerRings: View {
    let players: [CGRect]   // Vision-normalized, origin bottom-left
    let locked: CGRect?

    var body: some View {
        GeometryReader { geo in
            ForEach(Array(players.enumerated()), id: \.offset) { _, box in
                // skip the ring that overlaps the locked target's brackets
                if locked.map({ hypot($0.midX - box.midX, $0.midY - box.midY) > 0.06 }) ?? true {
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(Color.white.opacity(0.55), lineWidth: 1.2)
                        .frame(width: box.width * geo.size.width,
                               height: box.height * geo.size.height)
                        .position(x: box.midX * geo.size.width,
                                  y: (1 - box.midY) * geo.size.height)
                }
            }
        }
        .animation(.linear(duration: 0.12), value: players)
    }
}

// MARK: - AF-style corner brackets around the tracked subject

struct TargetBrackets: View {
    let box: CGRect  // Vision-normalized, origin bottom-left

    var body: some View {
        GeometryReader { geo in
            let r = CGRect(x: box.minX * geo.size.width,
                           y: (1 - box.maxY) * geo.size.height,
                           width: box.width * geo.size.width,
                           height: box.height * geo.size.height)
            let arm = min(r.width, r.height) * 0.28
            Path { p in
                for (cx, cy, dx, dy) in [(r.minX, r.minY, 1.0, 1.0), (r.maxX, r.minY, -1.0, 1.0),
                                         (r.minX, r.maxY, 1.0, -1.0), (r.maxX, r.maxY, -1.0, -1.0)] {
                    p.move(to: CGPoint(x: cx + dx * arm, y: cy))
                    p.addLine(to: CGPoint(x: cx, y: cy))
                    p.addLine(to: CGPoint(x: cx, y: cy + dy * arm))
                }
            }
            .stroke(Color.pla, lineWidth: 2.5)
        }
    }
}

// MARK: - camera-idiom record button: red core morphs to a square

struct RecordButton: View {
    let isRecording: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .stroke(.white, lineWidth: 3.5)
                    .frame(width: 66, height: 66)
                RoundedRectangle(cornerRadius: isRecording ? 6 : 27)
                    .fill(.red)
                    .frame(width: isRecording ? 30 : 54,
                           height: isRecording ? 30 : 54)
            }
            .animation(.spring(duration: 0.25), value: isRecording)
        }
    }
}

// MARK: - live timecode while recording

struct TimecodeView: View {
    let since: Date

    var body: some View {
        TimelineView(.periodic(from: since, by: 1)) { context in
            let s = Int(context.date.timeIntervalSince(since))
            HStack(spacing: 6) {
                Circle().fill(.red).frame(width: 8, height: 8)
                    .opacity(s % 2 == 0 ? 1 : 0.35)
                Text(String(format: "%02d:%02d", s / 60, s % 60))
                    .font(.system(size: 13, weight: .bold, design: .monospaced))
                    .foregroundColor(.white)
            }
        }
    }
}

/// UIKit wrapper for the AVCaptureVideoPreviewLayer. The host view re-frames
/// the layer in layoutSubviews — updateUIView does NOT run on layout, which
/// left the preview at zero size (black screen) before.
final class PreviewHostView: UIView {
    var previewLayer: AVCaptureVideoPreviewLayer? {
        didSet {
            oldValue?.removeFromSuperlayer()
            if let l = previewLayer {
                layer.addSublayer(l)
                setNeedsLayout()
            }
        }
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer?.frame = bounds
    }
}

struct CameraPreview: UIViewRepresentable {
    let layer: AVCaptureVideoPreviewLayer

    func makeUIView(context: Context) -> PreviewHostView {
        let view = PreviewHostView()
        view.previewLayer = layer
        return view
    }

    func updateUIView(_ view: PreviewHostView, context: Context) {
        if view.previewLayer !== layer { view.previewLayer = layer }
    }
}
