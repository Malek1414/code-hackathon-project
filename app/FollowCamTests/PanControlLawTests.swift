import XCTest
@testable import FollowCam

/// The controller's arithmetic, without a socket or a rig. Angles are servo
/// degrees (40…140, 90 = straight ahead); `centerX` is the subject's
/// Vision-normalized horizontal position (0 = left edge, 1 = right edge).
final class PanControlLawTests: XCTestCase {
    func testCenteredSubjectHoldsTheAngle() {
        XCTAssertEqual(PanController.step(angle: 90, centerX: 0.5, dt: 0.1, invert: false), 90)
        // Anywhere inside the deadband is "centered".
        let edge = 0.5 + PanController.deadband * 0.9
        XCTAssertEqual(PanController.step(angle: 73, centerX: edge, dt: 0.1, invert: false), 73)
    }

    func testOffCenterPansTowardTheSubjectAtARateProportionalToTheOffset() {
        // Full-frame error (0.5) at 180 deg/s for 0.1 s is 9 degrees.
        XCTAssertEqual(PanController.step(angle: 90, centerX: 1.0, dt: 0.1, invert: false), 99, accuracy: 1e-9)
        XCTAssertEqual(PanController.step(angle: 90, centerX: 0.0, dt: 0.1, invert: false), 81, accuracy: 1e-9)
        // Half the offset, half the rate.
        XCTAssertEqual(PanController.step(angle: 90, centerX: 0.75, dt: 0.1, invert: false), 94.5, accuracy: 1e-9)
    }

    func testInvertFlipsTheDirection() {
        XCTAssertEqual(PanController.step(angle: 90, centerX: 1.0, dt: 0.1, invert: true), 81, accuracy: 1e-9)
    }

    func testStepIsFrameRateIndependent() {
        // Three 10 ms steps land where one 30 ms step does.
        var a = 90.0
        for _ in 0..<3 { a = PanController.step(angle: a, centerX: 0.8, dt: 0.01, invert: false) }
        XCTAssertEqual(a, PanController.step(angle: 90, centerX: 0.8, dt: 0.03, invert: false), accuracy: 1e-9)
    }

    func testAngleIsClampedToTheLinkageRange() {
        XCTAssertEqual(PanController.step(angle: 139, centerX: 1.0, dt: 1, invert: false), PanController.maxAngle)
        XCTAssertEqual(PanController.step(angle: 41, centerX: 0.0, dt: 1, invert: false), PanController.minAngle)
        // Pinned at the stop, pushing further stays pinned.
        XCTAssertEqual(PanController.step(angle: PanController.maxAngle, centerX: 1.0, dt: 1, invert: false),
                       PanController.maxAngle)
    }
}
