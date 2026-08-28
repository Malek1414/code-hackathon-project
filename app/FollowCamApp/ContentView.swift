import AVFoundation
import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @StateObject private var pan = PanController()
    @State private var tracker = SubjectTracker()
    @State private var subjectBox: CGRect?   // Vision-normalized
    @State private var bridgeHost = "192.168.1.10"

    var body: some View {
        ZStack {
            CameraPreview(layer: camera.previewLayer)
                .ignoresSafeArea()
                .onTapGesture { location in
                    let roi = camera.normalizedRect(aroundLayerPoint: location)
                    tracker.select(roi: roi)
                }

            if let box = subjectBox {
                GeometryReader { geo in
                    // Vision: origin bottom-left, y up -> flip for SwiftUI
                    let r = CGRect(x: box.minX * geo.size.width,
                                   y: (1 - box.maxY) * geo.size.height,
                                   width: box.width * geo.size.width,
                                   height: box.height * geo.size.height)
                    Rectangle()
                        .path(in: r)
                        .stroke(Color.orange, lineWidth: 3)
                }
                .allowsHitTesting(false)
            }

            VStack {
                HStack {
                    TextField("laptop IP", text: $bridgeHost)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 160)
                    Button(pan.connected ? "linked" : "link rig") {
                        pan.connect(host: bridgeHost)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(pan.connected ? .green : .orange)
                    Toggle("inv", isOn: $pan.invert).frame(width: 80)
                    Spacer()
                    Text("\(Int(pan.angle))°")
                        .font(.system(.title2, design: .monospaced).bold())
                        .foregroundColor(.orange)
                }
                .padding()
                .background(.black.opacity(0.4))

                Spacer()

                HStack(spacing: 24) {
                    Button(tracker.isTracking ? "clear target" : "tap video to target") {
                        tracker.clear()
                        subjectBox = nil
                        pan.recenter()
                    }
                    .buttonStyle(.bordered)
                    .tint(.white)

                    Button {
                        camera.isRecording ? camera.stopRecording() : camera.startRecording()
                    } label: {
                        Circle()
                            .fill(camera.isRecording ? Color.gray : Color.red)
                            .frame(width: 64, height: 64)
                            .overlay(Circle().stroke(.white, lineWidth: 4))
                    }
                }
                .padding(.bottom, 32)
            }
        }
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
