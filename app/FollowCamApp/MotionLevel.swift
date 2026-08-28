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
            guard let m = motion else { return }
            // Portrait: gravity x tilts when the rig leans sideways.
            self?.rollDegrees = atan2(m.gravity.x, -m.gravity.y) * 180 / .pi
        }
    }

    func stop() {
        manager.stopDeviceMotionUpdates()
    }
}
