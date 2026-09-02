import XCTest
@testable import FollowCam

/// Bluetooth Heart Rate Measurement (0x2A37) packets as a strap sends them.
final class HeartRateParsingTests: XCTestCase {
    func testEightBitPacket() {
        XCTAssertEqual(HeartRateMonitor.parseHeartRate(Data([0x00, 72])), 72)
    }

    func testSixteenBitPacketIsLittleEndian() {
        // flags bit 0 set; 0x012C = 300
        XCTAssertEqual(HeartRateMonitor.parseHeartRate(Data([0x01, 0x2C, 0x01])), 300)
    }

    func testTrailingFieldsAreIgnored() {
        // 8-bit bpm followed by energy expended and an RR interval.
        XCTAssertEqual(HeartRateMonitor.parseHeartRate(Data([0x18, 155, 0x10, 0x00, 0x40, 0x03])), 155)
    }

    func testPacketsShorterThanTheirFlagsDeclareAreRejected() {
        XCTAssertNil(HeartRateMonitor.parseHeartRate(Data()))
        XCTAssertNil(HeartRateMonitor.parseHeartRate(Data([0x00])))
        // Declares 16-bit, carries one byte: this used to read past the end.
        XCTAssertNil(HeartRateMonitor.parseHeartRate(Data([0x01, 0x2C])))
    }

    func testSlicedDataIsReadFromItsOwnStart() {
        let framed = Data([0xFF, 0xFF, 0x00, 64])
        XCTAssertEqual(HeartRateMonitor.parseHeartRate(framed[2...]), 64)
    }
}
