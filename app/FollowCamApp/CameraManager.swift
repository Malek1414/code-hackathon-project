import AVFoundation
import UIKit

/// Owns the capture session, hands every frame to `onFrame`, and records
/// H.264 video to a temp file that is saved to Photos on stop.
final class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let previewLayer: AVCaptureVideoPreviewLayer
    var onFrame: ((CVPixelBuffer) -> Void)?

    @Published var isRecording = false

    private let queue = DispatchQueue(label: "followcam.camera")
    private var writer: AVAssetWriter?
    private var writerInput: AVAssetWriterInput?
    private var recordingURL: URL?
    private var sessionStarted = false

    override init() {
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        super.init()
    }

    private func configure() {
        session.beginConfiguration()
        session.sessionPreset = .hd1920x1080
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }
        if let conn = output.connection(with: .video) { conn.videoOrientation = .portrait }
        session.commitConfiguration()
    }

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { return }
            self.queue.async {
                if self.session.inputs.isEmpty { self.configure() }
                self.session.startRunning()
            }
        }
    }

    /// Convert a tap in preview-layer coordinates to a normalized ROI for Vision.
    func normalizedRect(aroundLayerPoint p: CGPoint, side: CGFloat = 120) -> CGRect {
        let layerRect = CGRect(x: p.x - side / 2, y: p.y - side / 2, width: side, height: side)
        return previewLayer.metadataOutputRectConverted(fromLayerRect: layerRect)
    }

    // MARK: recording

    func startRecording() {
        queue.async {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("followcam_\(Int(Date().timeIntervalSince1970)).mp4")
            guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mp4) else { return }
            let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: 1080,
                AVVideoHeightKey: 1920,
            ])
            input.expectsMediaDataInRealTime = true
            writer.add(input)
            self.writer = writer
            self.writerInput = input
            self.recordingURL = url
            self.sessionStarted = false
            writer.startWriting()
            DispatchQueue.main.async { self.isRecording = true }
        }
    }

    func stopRecording() {
        queue.async {
            guard let writer = self.writer, let url = self.recordingURL else { return }
            self.writerInput?.markAsFinished()
            writer.finishWriting {
                UISaveVideoAtPathToSavedPhotosAlbum(url.path, nil, nil, nil)
            }
            self.writer = nil
            self.writerInput = nil
            DispatchQueue.main.async { self.isRecording = false }
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        if let writer = writer, let input = writerInput {
            let ts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            if !sessionStarted {
                writer.startSession(atSourceTime: ts)
                sessionStarted = true
            }
            if input.isReadyForMoreMediaData { input.append(sampleBuffer) }
        }

        onFrame?(pixelBuffer)
    }
}
