import CoreGraphics
import CoreVideo
import Vision

/// Tracks one user-selected subject (the ball, or the person carrying it)
/// across frames with Vision. Coordinates are Vision-normalized (origin
/// bottom-left, y up).
final class SubjectTracker {
    private var request: VNTrackObjectRequest?
    private var handler = VNSequenceRequestHandler()
    private let humanRequest: VNDetectHumanRectanglesRequest = {
        let r = VNDetectHumanRectanglesRequest()
        r.upperBodyOnly = false
        return r
    }()

    var isTracking: Bool { request != nil }
    private(set) var frameIndex = 0   // incremented per track() call (camera queue)

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
        request = req
        handler = VNSequenceRequestHandler()
    }

    func clear() {
        request = nil
    }

    /// Returns the subject's bounding box for this frame, or nil if lost.
    func track(in pixelBuffer: CVPixelBuffer) -> CGRect? {
        frameIndex += 1
        guard let request = request else { return nil }
        do {
            try handler.perform([request], on: pixelBuffer)
        } catch {
            clear()
            return nil
        }
        guard let result = request.results?.first as? VNDetectedObjectObservation,
              result.confidence > 0.3 else {
            clear()
            return nil
        }
        request.inputObservation = result
        return result.boundingBox
    }
}
