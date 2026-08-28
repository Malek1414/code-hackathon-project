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
    private let deadband = 0.035          // |error| below this: subject is centered, hold
    private let rateDegPerSec = 180.0     // full-frame error (0.5) pans at 90 deg/s
    private let sendInterval = 1.0 / 15.0 // firmware slews at ~133 deg/s; 15 Hz is plenty
    private var lastSend = Date.distantPast
    private var lastUpdate: Date?
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
    /// Proportional rate control, frame-rate independent: a centered subject
    /// holds the angle (deadband), an off-center one pans toward it at a rate
    /// proportional to the offset — no per-frame integrator windup.
    func update(centerX: Double) {
        let now = Date()
        let dt = min(0.1, now.timeIntervalSince(lastUpdate ?? now))
        lastUpdate = now
        let error = (centerX - 0.5) * (invert ? -1 : 1)
        if abs(error) > deadband {
            let next = min(Self.maxAngle, max(Self.minAngle, angle + error * rateDegPerSec * dt))
            // publish only on real change: pinned at 40/140 (or inside the
            // deadband) the HUD must not re-render 60x/s for the same value
            if abs(next - angle) > 0.01 { angle = next }
        }
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
