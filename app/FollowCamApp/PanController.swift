import Foundation

/// Sends pan angles to the laptop bridge (software/pan_bridge.py) over a
/// WebSocket; the bridge forwards them to the Arduino as "A<angle>\n".
/// Control law: proportional on the subject's horizontal offset from frame
/// center, slew handled firmware-side.
///
/// The link is owned, not fire-and-forget: `connected` turns true only when
/// the socket has actually opened (the old version set it the moment
/// `resume()` was called, so the RIG lamp went green with no laptop in
/// sight), a ping every few seconds keeps a quiet socket alive, and a drop
/// reconnects with backoff until `disconnect()` is called.
final class PanController: NSObject, ObservableObject, URLSessionWebSocketDelegate {
    @Published var angle: Double = 90
    @Published var connected = false
    @Published var invert: Bool {
        didSet { UserDefaults.standard.set(invert, forKey: Self.invertKey) }
    }

    static let minAngle = 40.0, maxAngle = 140.0
    /// |error| below this: subject is centered, hold.
    static let deadband = 0.035
    /// Full-frame error (0.5) pans at 90 deg/s.
    static let rateDegPerSec = 180.0
    private static let invertKey = "pan.invert"

    private let sendInterval = 1.0 / 15.0 // firmware slews at ~133 deg/s; 15 Hz is plenty
    private let pingInterval: TimeInterval = 5
    private var lastSend = Date.distantPast
    private var lastUpdate: Date?
    private var task: URLSessionWebSocketTask?
    private lazy var session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    private var host: String?
    private var wantsLink = false
    private var reconnectPending = false
    private var attempts = 0
    private var pingTimer: Timer?

    override init() {
        invert = UserDefaults.standard.bool(forKey: Self.invertKey)
        super.init()
    }

    func connect(host: String) {
        wantsLink = true
        self.host = host
        attempts = 0
        open()
    }

    func disconnect() {
        wantsLink = false
        pingTimer?.invalidate()
        pingTimer = nil
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        connected = false
    }

    private func open() {
        reconnectPending = false
        task?.cancel(with: .goingAway, reason: nil)
        guard let host, let url = URL(string: "ws://\(host):8765") else { return }
        let t = session.webSocketTask(with: url)
        task = t
        t.resume()
        receiveLoop(t)
    }

    // MARK: control law

    /// One step of the controller, pure so it can be tested without a socket.
    /// `centerX` is the tracked subject's Vision-normalized center (0…1).
    /// Proportional rate control, frame-rate independent: a centered subject
    /// holds the angle (deadband), an off-center one pans toward it at a rate
    /// proportional to the offset — no per-frame integrator windup.
    static func step(angle: Double, centerX: Double, dt: Double, invert: Bool) -> Double {
        let error = (centerX - 0.5) * (invert ? -1 : 1)
        guard abs(error) > deadband else { return angle }
        return min(maxAngle, max(minAngle, angle + error * rateDegPerSec * dt))
    }

    func update(centerX: Double) {
        let now = Date()
        let dt = min(0.1, now.timeIntervalSince(lastUpdate ?? now))
        lastUpdate = now
        let next = Self.step(angle: angle, centerX: centerX, dt: dt, invert: invert)
        // publish only on real change: pinned at 40/140 (or inside the
        // deadband) the HUD must not re-render 60x/s for the same value
        if abs(next - angle) > 0.01 { angle = next }
        sendIfDue()
    }

    func recenter() {
        angle = 90
        sendIfDue(force: true)
    }

    private func sendIfDue(force: Bool = false) {
        guard let task, connected else { return }
        let now = Date()
        guard force || now.timeIntervalSince(lastSend) >= sendInterval else { return }
        lastSend = now
        task.send(.string("A\(Int(angle.rounded()))")) { [weak self] error in
            if error != nil { self?.dropped() }
        }
    }

    private func receiveLoop(_ t: URLSessionWebSocketTask) {
        t.receive { [weak self] result in
            switch result {
            case .success: self?.receiveLoop(t)
            case .failure: self?.dropped()
            }
        }
    }

    // MARK: link lifecycle (delegate callbacks arrive off-main)

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didOpenWithProtocol protocol: String?) {
        DispatchQueue.main.async {
            guard webSocketTask === self.task else { return }
            self.connected = true
            self.attempts = 0
            self.startPing()
            // Tell the rig where we are the moment the link is up.
            self.sendIfDue(force: true)
        }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        dropped()
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if error != nil { dropped() }
    }

    private func startPing() {
        pingTimer?.invalidate()
        pingTimer = Timer.scheduledTimer(withTimeInterval: pingInterval, repeats: true) { [weak self] _ in
            self?.task?.sendPing { error in
                if error != nil { self?.dropped() }
            }
        }
    }

    /// Every failure path lands here, possibly several times for one drop
    /// (close, task completion and a failed receive all fire). One reconnect
    /// is scheduled per drop, with backoff capped at 8 s.
    private func dropped() {
        DispatchQueue.main.async {
            self.connected = false
            self.pingTimer?.invalidate()
            self.pingTimer = nil
            guard self.wantsLink, !self.reconnectPending else { return }
            self.reconnectPending = true
            let delay = min(8.0, pow(2.0, Double(self.attempts)))
            self.attempts += 1
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
                guard let self, self.wantsLink, self.reconnectPending else { return }
                self.open()
            }
        }
    }
}
