import Foundation

/// Sends pan angles to the laptop bridge (software/pan_bridge.py) over a
/// WebSocket; the bridge forwards them to the Arduino as "A<angle>\n".
/// Control law: proportional on the subject's horizontal offset from frame
/// center, slew handled firmware-side.
final class PanController: ObservableObject {
    @Published var angle: Double = 90
    @Published var connected = false
    @Published var invert = false

    static let minAngle = 40.0, maxAngle = 140.0
    private let gainDegPerFrame = 6.0     // deg per full half-frame of error
    private let sendInterval = 1.0 / 15.0 // firmware slews at ~133 deg/s; 15 Hz is plenty
    private var lastSend = Date.distantPast
    private var task: URLSessionWebSocketTask?

    func connect(host: String) {
        disconnect()
        guard let url = URL(string: "ws://\(host):8765") else { return }
        let t = URLSession.shared.webSocketTask(with: url)
        task = t
        t.resume()
        receiveLoop(t)
        connected = true
    }

    func disconnect() {
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        connected = false
    }

    /// `centerX` is the tracked subject's Vision-normalized center (0…1).
    func update(centerX: Double) {
        let error = centerX - 0.5
        let delta = gainDegPerFrame * 2 * error * (invert ? -1 : 1)
        angle = min(Self.maxAngle, max(Self.minAngle, angle + delta))
        sendIfDue()
    }

    func recenter() {
        angle = 90
        sendIfDue(force: true)
    }

    private func sendIfDue(force: Bool = false) {
        guard let task = task else { return }
        let now = Date()
        guard force || now.timeIntervalSince(lastSend) >= sendInterval else { return }
        lastSend = now
        task.send(.string("A\(Int(angle.rounded()))")) { [weak self] error in
            if error != nil { DispatchQueue.main.async { self?.connected = false } }
        }
    }

    private func receiveLoop(_ t: URLSessionWebSocketTask) {
        t.receive { [weak self] result in
            switch result {
            case .success: self?.receiveLoop(t)
            case .failure: DispatchQueue.main.async { self?.connected = false }
            }
        }
    }
}
