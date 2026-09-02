import CoreMotion
import Foundation

/// Device roll from CoreMotion — a level tripod is the difference between
/// broadcast-looking pans and a tilted horizon. Portrait mount: roll ≈ 0
/// when the phone (and therefore the head) is level.
final class MotionLevel: ObservableObject {
    @Published var rollDegrees: Double = 0

    private let manager = CMMotionManager()

    var isLevel: Bool { abs(rollDegrees) < 1.0 }

    func start() {
        guard manager.isDeviceMotionAvailable else { return }
        manager.deviceMotionUpdateInterval = 1.0 / 15.0
        manager.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
            guard let self, let m = motion else { return }
            // Portrait: gravity x tilts when the rig leans sideways.
            let roll = atan2(m.gravity.x, -m.gravity.y) * 180 / .pi
            // Publish only a visible change. Sensor noise on a still tripod is
            // well under 0.2°, and every publish re-evaluates the whole HUD.
            if abs(roll - rollDegrees) >= 0.2 { rollDegrees = roll }
        }
    }

    func stop() {
        manager.stopDeviceMotionUpdates()
    }
}
