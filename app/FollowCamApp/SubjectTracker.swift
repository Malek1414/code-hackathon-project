import CoreGraphics
import CoreVideo
import Vision

/// Tracks one user-selected subject (the ball, or the person carrying it)
/// across frames with Vision. Coordinates are Vision-normalized (origin
/// bottom-left, y up).
final class SubjectTracker {
    private var request: VNTrackObjectRequest?
    private var handler = VNSequenceRequestHandler()

    var isTracking: Bool { request != nil }

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
