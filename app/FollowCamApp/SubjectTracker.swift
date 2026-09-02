import CoreGraphics
import CoreVideo
import Foundation
import Vision

/// Tracks one user-selected subject (the ball, or the person carrying it)
/// across frames with Vision. Coordinates are Vision-normalized (origin
/// bottom-left, y up).
///
/// Two threads touch this object: the vision queue calls `track` for every
/// frame, and the main thread calls `select`/`clear` from a tap or the
/// auto-reacquire. Swapping the request and the sequence handler while a
/// `perform` is in flight is a data race that shows up as a crash inside
/// Vision, so every access to that state goes through one lock.
final class SubjectTracker {
    private let lock = NSLock()
    private var request: VNTrackObjectRequest?
    private var handler = VNSequenceRequestHandler()
    private var frames = 0
    // Only ever used on the vision queue.
    private let humanRequest: VNDetectHumanRectanglesRequest = {
        let r = VNDetectHumanRectanglesRequest()
        r.upperBodyOnly = false
        return r
    }()

    var isTracking: Bool { lock.withLock { request != nil } }
    /// Incremented per `track()` call (vision queue).
    var frameIndex: Int { lock.withLock { frames } }

    /// Detect every person in the frame (full-body boxes, Vision-normalized).
    /// Cheap enough to run every few frames; feeds the on-screen player rings
    /// and gives auto-reacquire real candidates instead of a blind reseed.
    func humans(in pixelBuffer: CVPixelBuffer) -> [CGRect] {
        let h = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? h.perform([humanRequest])
        return (humanRequest.results ?? [])
            .filter { $0.confidence > 0.5 }
            .map { $0.boundingBox }
    }

    func select(roi: CGRect) {
        let observation = VNDetectedObjectObservation(boundingBox: roi)
        let req = VNTrackObjectRequest(detectedObjectObservation: observation)
        req.trackingLevel = .accurate
        lock.withLock {
            request = req
            handler = VNSequenceRequestHandler()
        }
    }

    func clear() {
        lock.withLock { request = nil }
    }

    /// Returns the subject's bounding box for this frame, or nil if lost.
    func track(in pixelBuffer: CVPixelBuffer) -> CGRect? {
        lock.lock()
        defer { lock.unlock() }
        frames += 1
        guard let request else { return nil }
        do {
            try handler.perform([request], on: pixelBuffer)
        } catch {
            self.request = nil
            return nil
        }
        guard let result = request.results?.first as? VNDetectedObjectObservation,
              result.confidence > 0.3 else {
            self.request = nil
            return nil
        }
        request.inputObservation = result
        return result.boundingBox
    }
}
