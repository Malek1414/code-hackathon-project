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
    @State private var tracker = SubjectTracker()
    @State private var subjectBox: CGRect?
    @State private var bridgeHost = "192.168.1.10"
    @State private var showSettings = false
    @State private var recordingStarted: Date?

    var body: some View {
        ZStack {
            CameraPreview(layer: camera.previewLayer)
                .ignoresSafeArea()
                .onTapGesture { location in
                    tracker.select(roi: camera.normalizedRect(aroundLayerPoint: location))
                }

            if let box = subjectBox {
                TargetBrackets(box: box).allowsHitTesting(false)
            }

            VStack(spacing: 0) {
                statusStrip
                PanScale(angle: pan.angle)
                    .frame(height: 44)
                    .padding(.horizontal, 24)
                Spacer()
                bottomBar
            }
        }
        .preferredColorScheme(.dark)
        .sheet(isPresented: $showSettings) { settingsSheet }
        .onAppear {
            camera.onFrame = { pixelBuffer in
                guard let box = tracker.track(in: pixelBuffer) else {
                    DispatchQueue.main.async { subjectBox = nil }
                    return
                }
                DispatchQueue.main.async {
                    subjectBox = box
                    pan.update(centerX: box.midX)
                }
            }
            camera.start()
        }
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
            HStack {
                if tracker.isTracking {
                    Button {
                        tracker.clear()
                        subjectBox = nil
                        pan.recenter()
                    } label: {
                        Text("CLEAR TARGET")
                            .font(.system(size: 12, weight: .semibold, design: .monospaced))
                            .foregroundColor(.pla)
                    }
                } else {
                    Text("TAP TO TARGET")
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundColor(.chrome)
                }
                Spacer()
            }
            .padding(.horizontal, 28)

            RecordButton(isRecording: camera.isRecording) {
                if camera.isRecording {
                    camera.stopRecording()
                    heart.endLog(stamp: Int(Date().timeIntervalSince1970))
                    recordingStarted = nil
                } else {
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
                    Button(whoop.connected ? "Connected — sync latest workout" : "Connect WHOOP") {
                        Task {
                            try? await whoop.connect()
                            _ = try? await WhoopClient.shared.exportLatestWorkout()
                        }
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
            let x: (Double) -> CGFloat = { w * ($0 - 40) / 100 }
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
                // needle + readout ride together
                Rectangle()
                    .fill(Color.pla)
                    .frame(width: 2, height: 16)
                    .position(x: x(angle), y: 8)
                Text("\(Int(angle))°")
                    .font(.system(size: 13, weight: .bold, design: .monospaced))
                    .foregroundColor(.pla)
                    .position(x: min(max(x(angle), 16), w - 16), y: 38)
            }
            .animation(.linear(duration: 0.08), value: angle)
        }
        .padding(.top, 6)
        .background(.black.opacity(0.55))
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

/// UIKit wrapper for the AVCaptureVideoPreviewLayer.
struct CameraPreview: UIViewRepresentable {
    let layer: AVCaptureVideoPreviewLayer

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        layer.frame = view.bounds
        view.layer.addSublayer(layer)
        return view
    }

    func updateUIView(_ view: UIView, context: Context) {
        layer.frame = view.bounds
    }
}
