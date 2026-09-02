import AVFoundation
import Photos
import UIKit

/// Owns the capture session, hands every frame to `onFrame`, and records
/// H.264 video (with the microphone, when allowed) to a temp file that is
/// saved to Photos on stop.
///
/// The recorder and the tracker share the camera, not a queue. Every sample
/// buffer is appended to the writer on the capture queue and only then
/// offered to Vision on its own queue — and only if Vision is idle. The old
/// version ran tracking synchronously inside the capture callback, so a slow
/// Vision frame made the output discard the next ones and the recording
/// stuttered exactly when the action was hardest to follow.
final class CameraManager: NSObject, ObservableObject,
                           AVCaptureVideoDataOutputSampleBufferDelegate,
                           AVCaptureAudioDataOutputSampleBufferDelegate {
    enum SaveState: Equatable {
        case idle, saving, saved
        case failed(String)
    }

    let session = AVCaptureSession()
    let previewLayer: AVCaptureVideoPreviewLayer
    /// Called on the vision queue with the newest frame. A frame that arrives
    /// while the previous call is still running is skipped, never queued.
    var onFrame: ((CVPixelBuffer) -> Void)?

    @Published var isRecording = false
    @Published var saveState: SaveState = .idle
    @Published var hasAudio = false

    private let queue = DispatchQueue(label: "followcam.camera")
    private let visionQueue = DispatchQueue(label: "followcam.vision", qos: .userInitiated)
    private var visionBusy = false          // guarded by `queue`
    private var audioAttached = false       // guarded by `queue`
    private var writer: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var audioInput: AVAssetWriterInput?
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

    // MARK: audio

    /// The microphone is asked for on the first Record tap, not at launch:
    /// that is the moment the request makes sense to the person reading it.
    /// Denied means the clip is silent, not that recording fails.
    private func ensureAudio(then: @escaping () -> Void) {
        queue.async {
            if self.audioAttached { then(); return }
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                self.queue.async {
                    if granted { self.attachAudio() }
                    then()
                }
            }
        }
    }

    /// On `queue`. Both outputs deliver on the same queue on purpose: the
    /// writer state below is then only ever touched from one place.
    private func attachAudio() {
        guard !audioAttached,
              let mic = AVCaptureDevice.default(for: .audio),
              let input = try? AVCaptureDeviceInput(device: mic) else { return }
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        guard session.canAddInput(input) else { return }
        session.addInput(input)
        let output = AVCaptureAudioDataOutput()
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else { return }
        session.addOutput(output)
        audioAttached = true
        DispatchQueue.main.async { self.hasAudio = true }
    }

    // MARK: recording

    func startRecording() {
        ensureAudio { [self] in beginWriting() }
    }

    /// On `queue`.
    private func beginWriting() {
        guard writer == nil else { return }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("followcam_\(Int(Date().timeIntervalSince1970))").appendingPathExtension("mp4")
        guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mp4) else {
            DispatchQueue.main.async { self.saveState = .failed("Could not create the recording file.") }
            return
        }
        let video = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: 1080,
            AVVideoHeightKey: 1920,
        ])
        video.expectsMediaDataInRealTime = true
        writer.add(video)
        if audioAttached {
            let audio = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVNumberOfChannelsKey: 1,
                AVSampleRateKey: 48_000,
                AVEncoderBitRateKey: 96_000,
            ])
            audio.expectsMediaDataInRealTime = true
            if writer.canAdd(audio) {
                writer.add(audio)
                audioInput = audio
            }
        }
        self.writer = writer
        videoInput = video
        recordingURL = url
        sessionStarted = false
        guard writer.startWriting() else {
            let reason = writer.error?.localizedDescription ?? "Could not start writing."
            self.writer = nil
            videoInput = nil
            audioInput = nil
            recordingURL = nil
            DispatchQueue.main.async { self.saveState = .failed(reason) }
            return
        }
        DispatchQueue.main.async {
            self.isRecording = true
            self.saveState = .idle
        }
    }

    func stopRecording() {
        queue.async {
            guard let writer = self.writer, let url = self.recordingURL else { return }
            let started = self.sessionStarted
            self.videoInput?.markAsFinished()
            self.audioInput?.markAsFinished()
            self.writer = nil
            self.videoInput = nil
            self.audioInput = nil
            self.recordingURL = nil
            DispatchQueue.main.async {
                self.isRecording = false
                self.saveState = .saving
            }
            guard started, writer.status == .writing else {
                writer.cancelWriting()
                try? FileManager.default.removeItem(at: url)
                let reason = writer.error?.localizedDescription ?? "No frames were recorded."
                DispatchQueue.main.async { self.saveState = .failed(reason) }
                return
            }
            writer.finishWriting { self.saveToPhotos(url) }
        }
    }

    /// The temp file is the only copy until Photos has it, so it is deleted
    /// only on a confirmed save. The old path handed the file to
    /// `UISaveVideoAtPathToSavedPhotosAlbum` with no completion: failures
    /// were silent, and every clip stayed in tmp — a gigabyte per game.
    private func saveToPhotos(_ url: URL) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async {
                    self.saveState = .failed("Photos access was denied — the clip is in the app's tmp folder.")
                }
                return
            }
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
            }) { ok, error in
                if ok { try? FileManager.default.removeItem(at: url) }
                DispatchQueue.main.async {
                    self.saveState = ok ? .saved : .failed(error?.localizedDescription ?? "Could not save to Photos.")
                }
            }
        }
    }

    // MARK: capture callback (both outputs, on `queue`)

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        if output is AVCaptureAudioDataOutput {
            // Audio only joins once the first video frame has opened the
            // session; a sample before that has nothing to be timed against.
            if let writer, writer.status == .writing, sessionStarted,
               let audioInput, audioInput.isReadyForMoreMediaData {
                audioInput.append(sampleBuffer)
            }
            return
        }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        if let writer, writer.status == .writing, let videoInput {
            let ts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            if !sessionStarted {
                writer.startSession(atSourceTime: ts)
                sessionStarted = true
            }
            if videoInput.isReadyForMoreMediaData { videoInput.append(sampleBuffer) }
        }

        // Newest frame wins. Vision keeps at most one frame in flight, so a
        // slow detection costs the tracker a frame, never the recording.
        guard let onFrame, !visionBusy else { return }
        visionBusy = true
        visionQueue.async { [weak self] in
            onFrame(pixelBuffer)
            self?.queue.async { self?.visionBusy = false }
        }
    }
}
