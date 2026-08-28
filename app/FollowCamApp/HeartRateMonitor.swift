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

    private var central: CBCentralManager!
    private var peripheral: CBPeripheral?
    private let hrService = CBUUID(string: "180D")
    private let hrMeasurement = CBUUID(string: "2A37")

    private var log: [(Double, Int)] = []
    private var logging = false

    override init() {
        super.init()
        central = CBCentralManager(delegate: self, queue: nil)
    }

    func startScan() {
        guard central.state == .poweredOn else { return }
        scanning = true
        central.scanForPeripherals(withServices: [hrService])
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
        if central.state == .poweredOn && scanning { startScan() }
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
        guard let data = characteristic.value, data.count >= 2 else { return }
        // HR Measurement: flag bit 0 selects uint8 vs uint16 bpm
        let value = (data[0] & 0x01) == 0
            ? Int(data[1])
            : Int(UInt16(data[1]) | (UInt16(data[2]) << 8))
        DispatchQueue.main.async {
            self.bpm = value
            if self.logging { self.log.append((Date().timeIntervalSince1970, value)) }
        }
    }
}
