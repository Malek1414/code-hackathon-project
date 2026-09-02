import CoreBluetooth
import Foundation

/// Live heart rate over Bluetooth LE using the standard Heart Rate service
/// (0x180D). WHOOP's "Heart Rate Broadcast" mode advertises exactly this
/// profile, so no WHOOP API/auth is needed for LIVE data — enable broadcast
/// in the WHOOP app and this picks it up (also works with any chest strap).
/// While recording, every reading is logged as (unix_ts, bpm) so it can be
/// joined with the video timeline afterwards (ml/correlate_hr.py).
final class HeartRateMonitor: NSObject, ObservableObject, CBCentralManagerDelegate, CBPeripheralDelegate {
    @Published var bpm: Int?
    @Published var scanning = false

    // Created on the first tap of HR, not at launch: creating the central is
    // what triggers the Bluetooth permission prompt, and a prompt that
    // appears before the user asked for anything gets denied.
    private var central: CBCentralManager?
    private var peripheral: CBPeripheral?
    /// True from the first `startScan` until `stop`: a dropped strap is
    /// re-scanned for rather than left showing its last reading forever.
    private var wantsStrap = false
    private let hrService = CBUUID(string: "180D")
    private let hrMeasurement = CBUUID(string: "2A37")

    private var log: [(Double, Int)] = []
    private var logging = false

    var isActive: Bool { wantsStrap }

    func startScan() {
        wantsStrap = true
        scanning = true
        if central == nil {
            // The scan itself starts in centralManagerDidUpdateState once the
            // radio reports poweredOn; calling scanForPeripherals before that
            // is a no-op, which is why the first tap used to do nothing.
            central = CBCentralManager(delegate: self, queue: nil)
            return
        }
        scanIfPossible()
    }

    func stop() {
        wantsStrap = false
        scanning = false
        central?.stopScan()
        if let peripheral { central?.cancelPeripheralConnection(peripheral) }
        peripheral = nil
        bpm = nil
    }

    private func scanIfPossible() {
        guard let central, central.state == .poweredOn, wantsStrap else { return }
        central.scanForPeripherals(withServices: [hrService])
    }

    /// Bluetooth Heart Rate Measurement (0x2A37): flag bit 0 selects a uint8
    /// or uint16 bpm right after the flags byte. Anything else in the packet
    /// (energy expended, RR intervals) follows and is ignored. Returns nil
    /// for a packet too short for the format its own flags declare — a
    /// truncated 16-bit packet used to be read past its end.
    static func parseHeartRate(_ data: Data) -> Int? {
        let bytes = [UInt8](data)
        guard let flags = bytes.first else { return nil }
        if flags & 0x01 == 0 {
            return bytes.count >= 2 ? Int(bytes[1]) : nil
        }
        return bytes.count >= 3 ? Int(UInt16(bytes[1]) | (UInt16(bytes[2]) << 8)) : nil
    }

    // MARK: logging, aligned with recording start/stop

    func beginLog() {
        log.removeAll()
        logging = true
    }

    /// Ends the log and writes hr_<stamp>.csv to the app's Documents folder.
    func endLog(stamp: Int) {
        logging = false
        guard !log.isEmpty else { return }
        let lines = ["unix_ts,bpm"] + log.map { String(format: "%.2f,%d", $0.0, $0.1) }
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("hr_\(stamp).csv")
        try? lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
    }

    // MARK: CoreBluetooth

    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            scanIfPossible()
        default:
            // Radio off, unauthorized or resetting: the last reading is stale.
            bpm = nil
            scanning = false
        }
    }

    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                        advertisementData: [String: Any], rssi RSSI: NSNumber) {
        self.peripheral = peripheral
        central.stopScan()
        scanning = false
        central.connect(peripheral)
    }

    func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        peripheral.delegate = self
        peripheral.discoverServices([hrService])
    }

    func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        self.peripheral = nil
        if wantsStrap {
            scanning = true
            scanIfPossible()
        }
    }

    /// A strap that walks out of range must not leave its last bpm on the
    /// HUD as if it were live. Clear it and go back to scanning.
    func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        self.peripheral = nil
        bpm = nil
        if wantsStrap {
            scanning = true
            scanIfPossible()
        }
    }

    func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        guard let service = peripheral.services?.first(where: { $0.uuid == hrService }) else { return }
        peripheral.discoverCharacteristics([hrMeasurement], for: service)
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        guard let ch = service.characteristics?.first(where: { $0.uuid == hrMeasurement }) else { return }
        peripheral.setNotifyValue(true, for: ch)
    }

    func peripheral(_ peripheral: CBPeripheral,
                    didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard let data = characteristic.value, let value = Self.parseHeartRate(data) else { return }
        DispatchQueue.main.async {
            self.bpm = value
            if self.logging { self.log.append((Date().timeIntervalSince1970, value)) }
        }
    }
}
