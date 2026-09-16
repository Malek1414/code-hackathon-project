# Multi-View Synced Recording Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Several iPhones record one game with a shared clock, upload their clips after the game, and the pipeline aligns the views and fuses the ball across them, reporting how much of the game at least one view saw the ball.

**Architecture:** Three independent pieces joined by files. (1) iOS: a session coordinator over Multipeer Connectivity gives every phone a host-clock offset, the recorder stamps that into a sidecar JSON next to the clip, and a background uploader ships clip plus sidecar. (2) `ingest/`: a small FastAPI service on the Spark that stores uploads under `data/sessions/<id>/` and starts the pipeline when all views are in. (3) `vision/multiview/`: align (sidecar then audio), run the existing single-view tracker and stats per view, fuse in time (best view per 100 ms slot with hysteresis, union of shot events), write coverage and a report page. Every session is training data for the Spark plan.

**Tech Stack:** Swift 5 / SwiftUI / MultipeerConnectivity / AVFoundation / background URLSession (iOS 17, Xcode, xcodegen); Python 3.14 in `ml/.venv` with FastAPI, uvicorn, httpx, numpy, PyYAML, imageio-ffmpeg; the existing `vision/track` and `vision/stats` modules.

Spec: `docs/superpowers/specs/2026-09-16-spark-training-and-multiview-design.md`, Part B.

## Global Constraints

- Sidecar schema (spec B2), exact keys: `schema, session, view, device, host_offset_ms, host_offset_err_ms, record_start_host_ms, fps, width, height, orientation, label`. File name `<view>.followcam.json` next to `<view>.mp4`.
- Session manifest `session.json`: `schema, session, team, date, primary, views: [{view, label}]`.
- Time inside the pipeline is primary-view milliseconds. Slot length 100 ms. Ball counts if conf at or above 0.35. A best-view switch needs the challenger to win by 0.1 for 3 consecutive slots. Shot events dedupe within 1.5 s; the verdict comes from the view with the largest hoop box.
- Audio refinement window is plus or minus 500 ms around the sidecar offset; a refinement more than 300 ms from the sidecar is rejected and the view is flagged.
- One bearer token per team in `ingest/teams.yaml` (gitignored); `ingest/teams.example.yaml` is committed.
- Python: `ml/.venv/bin/python`. Tests: `ml/.venv/bin/python -m pytest ingest/tests vision/multiview/tests -q`. iOS tests: `cd app && xcodebuild test -scheme FollowCam -destination 'platform=iOS Simulator,name=iPhone 17 Pro'`.
- After adding Swift files run `cd app && xcodegen generate` (the `.xcodeproj` is committed).
- The rig link, HR strap and tap-to-track in the app stay untouched.
- Commit after every task; messages end with `Co-Authored-By: WOZCODE <contact@withwoz.com>`.

---

## File map

| Path | Responsibility |
|---|---|
| `app/FollowCamApp/Session/ClockSync.swift` | pure: NTP-style offset from round trips, median, error estimate |
| `app/FollowCamApp/Session/SessionModels.swift` | `ViewSidecar`, `SessionManifest`, `SessionMessage` (Codable) |
| `app/FollowCamApp/Session/SessionCoordinator.swift` | Multipeer host/join, ping loop, record/stop broadcast |
| `app/FollowCamApp/Session/SessionStore.swift` | Documents/Sessions/<id>/ files: clip copy, sidecar, manifest, listing |
| `app/FollowCamApp/Session/Uploader.swift` | background URLSession uploads with per-file state |
| `app/FollowCamApp/Session/SessionsView.swift` | Sessions screen: create/join, members, record state, upload state |
| `app/FollowCamApp/CameraManager.swift` | records host start time, hands the finished file to `SessionStore` before Photos |
| `app/FollowCamApp/ContentView.swift` | a Sessions button in the status strip; record button drives the session when one is active |
| `app/FollowCamTests/ClockSyncTests.swift`, `SessionModelsTests.swift` | pure-logic tests |
| `ingest/app.py`, `ingest/storage.py`, `ingest/watcher.py`, `ingest/teams.example.yaml`, `ingest/requirements.txt`, `ingest/tests/test_api.py` | upload service |
| `vision/multiview/__init__.py`, `align.py`, `detect.py`, `fuse.py`, `report.py`, `run.py`, `tests/` | pipeline |
| `Makefile` | `multiview-test`, `multiview RUN=<session>`, `ingest-serve` |
| `docs/MULTIVIEW.md` | how to record, upload, run |

---

### Task 1: Clock sync and session models (iOS, pure logic)

**Files:**
- Create: `app/FollowCamApp/Session/ClockSync.swift`, `app/FollowCamApp/Session/SessionModels.swift`, `app/FollowCamTests/ClockSyncTests.swift`, `app/FollowCamTests/SessionModelsTests.swift`

**Interfaces:**
- Produces: `struct ClockSync { static func offset(sent t0: Double, hostReceived t1: Double, hostSent t2: Double, received t3: Double) -> (offsetMs: Double, rttMs: Double) }`, `struct OffsetEstimate { let offsetMs: Double; let errMs: Double; let samples: Int }`, `static func estimate(_ samples: [(offsetMs: Double, rttMs: Double)]) -> OffsetEstimate?` (median offset of the lowest-RTT half, err = half the median RTT of those). `struct ViewSidecar: Codable` with the spec's keys, `struct SessionManifest: Codable`, `enum SessionMessage: Codable { case hello(label: String), ping(t0: Double), pong(t0: Double, t1: Double, t2: Double), record(startHostMs: Double), stop }` plus `static func encode(_:) -> Data` / `decode(_:) -> SessionMessage?`. Host time is milliseconds since 1970 as `Double`.

- [ ] **Step 1: Write the failing tests**

```swift
// app/FollowCamTests/ClockSyncTests.swift
import XCTest
@testable import FollowCam

final class ClockSyncTests: XCTestCase {
    func testOffsetIsNtpFormula() {
        // host clock is 100 ms ahead; 20 ms each way
        let r = ClockSync.offset(sent: 1000, hostReceived: 1120, hostSent: 1125, received: 1045)
        XCTAssertEqual(r.offsetMs, 100, accuracy: 1e-9)
        XCTAssertEqual(r.rttMs, 40, accuracy: 1e-9)
    }

    func testEstimateUsesLowestRttHalfAndMedian() {
        let samples: [(offsetMs: Double, rttMs: Double)] = [
            (100, 40), (101, 42), (99, 38), (150, 400), (50, 380), (100, 41)]
        let e = ClockSync.estimate(samples)!
        XCTAssertEqual(e.offsetMs, 100, accuracy: 0.5)   // the two outliers ride on slow round trips
        XCTAssertEqual(e.samples, 3)
        XCTAssertLessThan(e.errMs, 25)
    }

    func testEstimateNeedsAtLeastOneSample() {
        XCTAssertNil(ClockSync.estimate([]))
    }
}
```

```swift
// app/FollowCamTests/SessionModelsTests.swift
import XCTest
@testable import FollowCam

final class SessionModelsTests: XCTestCase {
    func testSidecarUsesSpecKeys() throws {
        let s = ViewSidecar(session: "S", view: "V", device: "iPhone16,1", hostOffsetMs: 12.4,
                            hostOffsetErrMs: 3.1, recordStartHostMs: 1758030000123, fps: 30,
                            width: 1080, height: 1920, orientation: "portrait", label: "baseline left")
        let json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(s)) as! [String: Any]
        XCTAssertEqual(Set(json.keys), ["schema", "session", "view", "device", "host_offset_ms",
                                        "host_offset_err_ms", "record_start_host_ms", "fps", "width",
                                        "height", "orientation", "label"])
        XCTAssertEqual(json["schema"] as? Int, 1)
        XCTAssertEqual(json["record_start_host_ms"] as? Double, 1758030000123)
    }

    func testMessagesRoundTrip() {
        let msgs: [SessionMessage] = [.hello(label: "left"), .ping(t0: 1), .pong(t0: 1, t1: 2, t2: 3),
                                      .record(startHostMs: 99), .stop]
        for m in msgs {
            XCTAssertEqual(SessionMessage.decode(SessionMessage.encode(m)), m)
        }
        XCTAssertNil(SessionMessage.decode(Data("junk".utf8)))
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd app && xcodegen generate && xcodebuild test -scheme FollowCam -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | tail -20`
Expected: build FAILS with `cannot find 'ClockSync' in scope` (and the model types).

- [ ] **Step 3: Implement the two files**

```swift
// app/FollowCamApp/Session/ClockSync.swift
import Foundation

/// NTP-style clock offset of this phone to the session host. All times are
/// milliseconds since 1970 on the respective phone's clock.
enum ClockSync {
    struct OffsetEstimate: Equatable {
        let offsetMs: Double
        let errMs: Double
        let samples: Int
    }

    /// t0: we sent the ping; t1: host received it; t2: host sent the pong; t3: we received it.
    static func offset(sent t0: Double, hostReceived t1: Double, hostSent t2: Double,
                       received t3: Double) -> (offsetMs: Double, rttMs: Double) {
        (((t1 - t0) + (t2 - t3)) / 2, (t3 - t0) - (t2 - t1))
    }

    /// Median offset over the lowest-RTT half of the samples; error is half
    /// their median RTT (the asymmetry we cannot see).
    static func estimate(_ samples: [(offsetMs: Double, rttMs: Double)]) -> OffsetEstimate? {
        guard !samples.isEmpty else { return nil }
        let byRtt = samples.sorted { $0.rttMs < $1.rttMs }
        let best = Array(byRtt.prefix(max(1, byRtt.count / 2)))
        let offsets = best.map(\.offsetMs).sorted()
        let rtts = best.map(\.rttMs).sorted()
        return OffsetEstimate(offsetMs: offsets[offsets.count / 2],
                              errMs: rtts[rtts.count / 2] / 2,
                              samples: best.count)
    }

    static var nowMs: Double { Date().timeIntervalSince1970 * 1000 }
}
```

```swift
// app/FollowCamApp/Session/SessionModels.swift
import Foundation

/// Written next to every clip as <view>.followcam.json (spec B2).
struct ViewSidecar: Codable, Equatable {
    var schema = 1
    let session: String
    let view: String
    let device: String
    let hostOffsetMs: Double
    let hostOffsetErrMs: Double
    let recordStartHostMs: Double
    let fps: Int
    let width: Int
    let height: Int
    let orientation: String
    let label: String

    enum CodingKeys: String, CodingKey {
        case schema, session, view, device, fps, width, height, orientation, label
        case hostOffsetMs = "host_offset_ms"
        case hostOffsetErrMs = "host_offset_err_ms"
        case recordStartHostMs = "record_start_host_ms"
    }
}

struct SessionManifest: Codable, Equatable {
    struct ViewEntry: Codable, Equatable {
        let view: String
        let label: String
    }
    var schema = 1
    let session: String
    let team: String
    let date: String          // ISO 8601, host's start time
    let primary: String       // view id
    var views: [ViewEntry]
}

/// Messages on the Multipeer session. The host answers pings; joiners follow record/stop.
enum SessionMessage: Codable, Equatable {
    case hello(label: String)
    case ping(t0: Double)
    case pong(t0: Double, t1: Double, t2: Double)
    case record(startHostMs: Double)
    case stop

    static func encode(_ m: SessionMessage) -> Data { (try? JSONEncoder().encode(m)) ?? Data() }
    static func decode(_ d: Data) -> SessionMessage? { try? JSONDecoder().decode(SessionMessage.self, from: d) }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd app && xcodegen generate && xcodebuild test -scheme FollowCam -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | grep -E "Test Suite|passed|failed" | tail -5`
Expected: `ClockSyncTests` and `SessionModelsTests` pass, existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add app/FollowCamApp/Session app/FollowCamTests/ClockSyncTests.swift app/FollowCamTests/SessionModelsTests.swift app/FollowCam.xcodeproj/project.pbxproj
git commit -m "app: clock sync estimator and session models

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

---

### Task 2: Session coordinator over Multipeer

**Files:**
- Create: `app/FollowCamApp/Session/SessionCoordinator.swift`
- Modify: `app/project.yml` (add `INFOPLIST_KEY_NSBonjourServices` and a local-network usage string already exists)

**Interfaces:**
- Consumes: `ClockSync`, `SessionMessage`.
- Produces: `final class SessionCoordinator: NSObject, ObservableObject` with `@Published var role: Role` (`.none, .host, .joiner`), `@Published var sessionId: String?`, `@Published var code: String` (4 letters shown on the host), `@Published var members: [Member]` (`id, label, offsetMs, errMs`), `@Published var offset: ClockSync.OffsetEstimate?` (joiner), `@Published var recordStartHostMs: Double?` (set on everyone when the host records), `var viewId: String` (per install, persisted), `var label: String` (persisted), `func host(team: String)`, `func join()`, `func leave()`, `func startRecording()` (host only, broadcasts `.record(startHostMs: nowMs + 500)`), `func stopRecording()` (host only), `var onRecord: ((Double) -> Void)?`, `var onStop: (() -> Void)?`. `func hostNowMs() -> Double` returns this phone's best estimate of host time (own clock plus offset; the host returns its own clock).

Deviation from the spec's QR code: the host shows a 4-letter code and advertises it in Multipeer `discoveryInfo`; joiners browse, see hosts nearby with their code, and tap the matching one. Same trust model, no second camera session for a QR scan while the recorder owns the camera.

- [ ] **Step 1: Add the Bonjour service to `project.yml`**

Under `targets.FollowCam.settings.base` add:

```yaml
        INFOPLIST_KEY_NSBonjourServices: "_followcam-sess._tcp,_followcam-sess._udp"
```
(`NSLocalNetworkUsageDescription` is already set.)

- [ ] **Step 2: Implement `SessionCoordinator.swift`**

```swift
// app/FollowCamApp/Session/SessionCoordinator.swift
import Foundation
import MultipeerConnectivity
import UIKit

/// One phone hosts a recording session; the others join over Multipeer.
/// Joiners ping the host every 2 s and keep a clock offset. The host's
/// record/stop are broadcast so every phone stamps the same host time.
final class SessionCoordinator: NSObject, ObservableObject {
    enum Role { case none, host, joiner }
    struct Member: Identifiable, Equatable {
        let id: String        // peer display name = view id
        var label: String
        var offsetMs: Double?
        var errMs: Double?
    }
    struct NearbyHost: Identifiable, Equatable {
        let id: MCPeerID
        let code: String
        let team: String
        let session: String
    }

    static let serviceType = "followcam-sess"

    @Published var role: Role = .none
    @Published var sessionId: String?
    @Published var team: String = ""
    @Published var code: String = ""
    @Published var members: [Member] = []
    @Published var nearby: [NearbyHost] = []
    @Published var offset: ClockSync.OffsetEstimate?
    @Published var recordStartHostMs: Double?
    @Published var label: String {
        didSet { UserDefaults.standard.set(label, forKey: "session.label") }
    }
    let viewId: String

    var onRecord: ((Double) -> Void)?
    var onStop: (() -> Void)?

    private let peer: MCPeerID
    private lazy var session = MCSession(peer: peer, securityIdentity: nil, encryptionPreference: .required)
    private var advertiser: MCNearbyServiceAdvertiser?
    private var browser: MCNearbyServiceBrowser?
    private var pingTimer: Timer?
    private var samples: [(offsetMs: Double, rttMs: Double)] = []
    private var hostPeer: MCPeerID?

    override init() {
        let defaults = UserDefaults.standard
        if let id = defaults.string(forKey: "session.viewId") {
            viewId = id
        } else {
            viewId = UUID().uuidString.lowercased()
            defaults.set(viewId, forKey: "session.viewId")
        }
        label = defaults.string(forKey: "session.label") ?? UIDevice.current.name
        peer = MCPeerID(displayName: viewId)
        super.init()
        session.delegate = self
    }

    // MARK: host

    func host(team: String) {
        leave()
        self.team = team
        sessionId = UUID().uuidString.lowercased()
        code = String((0..<4).map { _ in "ABCDEFGHJKLMNPQRSTUVWXYZ".randomElement()! })
        role = .host
        members = [Member(id: viewId, label: label, offsetMs: 0, errMs: 0)]
        let adv = MCNearbyServiceAdvertiser(peer: peer, discoveryInfo: ["code": code, "team": team, "session": sessionId!],
                                            serviceType: Self.serviceType)
        adv.delegate = self
        adv.startAdvertisingPeer()
        advertiser = adv
    }

    func startRecording() {
        guard role == .host else { return }
        let start = ClockSync.nowMs + 500
        recordStartHostMs = start
        broadcast(.record(startHostMs: start))
        onRecord?(start)
    }

    func stopRecording() {
        guard role == .host else { return }
        broadcast(.stop)
        recordStartHostMs = nil
        onStop?()
    }

    // MARK: joiner

    func browse() {
        leave()
        role = .joiner
        let b = MCNearbyServiceBrowser(peer: peer, serviceType: Self.serviceType)
        b.delegate = self
        b.startBrowsingForPeers()
        browser = b
    }

    func join(_ host: NearbyHost) {
        guard let browser else { return }
        hostPeer = host.id
        code = host.code
        team = host.team
        sessionId = host.session
        browser.invitePeer(host.id, to: session, withContext: Data(label.utf8), timeout: 15)
    }

    func leave() {
        pingTimer?.invalidate(); pingTimer = nil
        advertiser?.stopAdvertisingPeer(); advertiser = nil
        browser?.stopBrowsingForPeers(); browser = nil
        session.disconnect()
        role = .none; sessionId = nil; code = ""; members = []; nearby = []
        offset = nil; recordStartHostMs = nil; samples = []; hostPeer = nil
    }

    func hostNowMs() -> Double {
        ClockSync.nowMs + (role == .joiner ? (offset?.offsetMs ?? 0) : 0)
    }

    // MARK: wire

    private func broadcast(_ m: SessionMessage) {
        guard !session.connectedPeers.isEmpty else { return }
        try? session.send(SessionMessage.encode(m), toPeers: session.connectedPeers, with: .reliable)
    }

    private func send(_ m: SessionMessage, to peer: MCPeerID) {
        try? session.send(SessionMessage.encode(m), toPeers: [peer], with: .reliable)
    }

    private func startPinging() {
        pingTimer?.invalidate()
        pingTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            guard let self, let hostPeer else { return }
            self.send(.ping(t0: ClockSync.nowMs), to: hostPeer)
        }
        pingTimer?.fire()
    }

    private func handle(_ m: SessionMessage, from peer: MCPeerID) {
        switch m {
        case .hello(let label):
            if let i = members.firstIndex(where: { $0.id == peer.displayName }) {
                members[i].label = label
            } else {
                members.append(Member(id: peer.displayName, label: label, offsetMs: nil, errMs: nil))
            }
        case .ping(let t0):
            let t1 = ClockSync.nowMs
            send(.pong(t0: t0, t1: t1, t2: ClockSync.nowMs), to: peer)
        case .pong(let t0, let t1, let t2):
            let s = ClockSync.offset(sent: t0, hostReceived: t1, hostSent: t2, received: ClockSync.nowMs)
            samples.append(s)
            if samples.count > 16 { samples.removeFirst() }
            offset = ClockSync.estimate(samples)
            if let offset, let hostPeer {
                send(.hello(label: "\(label)|\(offset.offsetMs)|\(offset.errMs)"), to: hostPeer)
            }
        case .record(let start):
            recordStartHostMs = start
            onRecord?(start)
        case .stop:
            recordStartHostMs = nil
            onStop?()
        }
    }
}

extension SessionCoordinator: MCSessionDelegate {
    func session(_ session: MCSession, peer peerID: MCPeerID, didChange state: MCSessionState) {
        DispatchQueue.main.async {
            switch (self.role, state) {
            case (.joiner, .connected):
                self.send(.hello(label: self.label), to: peerID)
                self.startPinging()
            case (.host, .connected):
                if !self.members.contains(where: { $0.id == peerID.displayName }) {
                    self.members.append(Member(id: peerID.displayName, label: peerID.displayName, offsetMs: nil, errMs: nil))
                }
            case (.host, .notConnected):
                self.members.removeAll { $0.id == peerID.displayName }
            case (.joiner, .notConnected):
                self.pingTimer?.invalidate(); self.pingTimer = nil
                self.offset = nil
            default: break
            }
        }
    }

    func session(_ session: MCSession, didReceive data: Data, fromPeer peerID: MCPeerID) {
        guard let m = SessionMessage.decode(data) else { return }
        DispatchQueue.main.async {
            // joiners report "label|offset|err" through hello so the host can show them
            if case .hello(let raw) = m, self.role == .host, raw.contains("|") {
                let parts = raw.split(separator: "|").map(String.init)
                if let i = self.members.firstIndex(where: { $0.id == peerID.displayName }), parts.count == 3 {
                    self.members[i].label = parts[0]
                    self.members[i].offsetMs = Double(parts[1])
                    self.members[i].errMs = Double(parts[2])
                }
                return
            }
            self.handle(m, from: peerID)
        }
    }

    func session(_ session: MCSession, didReceive stream: InputStream, withName streamName: String, fromPeer peerID: MCPeerID) {}
    func session(_ session: MCSession, didStartReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, with progress: Progress) {}
    func session(_ session: MCSession, didFinishReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, at localURL: URL?, withError error: Error?) {}
}

extension SessionCoordinator: MCNearbyServiceAdvertiserDelegate {
    func advertiser(_ advertiser: MCNearbyServiceAdvertiser, didReceiveInvitationFromPeer peerID: MCPeerID,
                    withContext context: Data?, invitationHandler: @escaping (Bool, MCSession?) -> Void) {
        invitationHandler(true, session)
    }
}

extension SessionCoordinator: MCNearbyServiceBrowserDelegate {
    func browser(_ browser: MCNearbyServiceBrowser, foundPeer peerID: MCPeerID, withDiscoveryInfo info: [String: String]?) {
        DispatchQueue.main.async {
            let h = NearbyHost(id: peerID, code: info?["code"] ?? "????", team: info?["team"] ?? "",
                               session: info?["session"] ?? "")
            if !self.nearby.contains(h) { self.nearby.append(h) }
        }
    }

    func browser(_ browser: MCNearbyServiceBrowser, lostPeer peerID: MCPeerID) {
        DispatchQueue.main.async { self.nearby.removeAll { $0.id == peerID } }
    }
}
```

- [ ] **Step 3: Build**

Run: `cd app && xcodegen generate && xcodebuild -scheme FollowCam -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build 2>&1 | tail -3`
Expected: `** BUILD SUCCEEDED **`

- [ ] **Step 4: Commit**

```bash
git add app/project.yml app/FollowCamApp/Session/SessionCoordinator.swift app/FollowCam.xcodeproj/project.pbxproj
git commit -m "app: Multipeer session coordinator with clock offsets and record broadcast

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

---

### Task 3: Session store and sidecar written by the recorder

**Files:**
- Create: `app/FollowCamApp/Session/SessionStore.swift`, `app/FollowCamTests/SessionStoreTests.swift`
- Modify: `app/FollowCamApp/CameraManager.swift` (lines 121-200: `startRecording`, `beginWriting`, `stopRecording`)

**Interfaces:**
- Consumes: `ViewSidecar`, `SessionManifest`, `SessionCoordinator.hostNowMs()`.
- Produces: `struct RecordingContext { let session: String; let view: String; let label: String; let hostOffsetMs: Double; let hostOffsetErrMs: Double; var recordStartHostMs: Double }`; `final class SessionStore` with `static let root: URL` (Documents/Sessions), `func keep(clip: URL, context: RecordingContext, fps: Int, width: Int, height: Int) throws -> (clip: URL, sidecar: URL)` (copies the clip to `Sessions/<session>/<view>.mp4` and writes the sidecar), `func writeManifest(_ m: SessionManifest) throws -> URL`, `func sessions() -> [StoredSession]` (`id, clips: [(clip: URL, sidecar: URL)], manifest: URL?`), `func remove(session: String)`. `CameraManager` gains `var recordingContext: RecordingContext?` and `var onKept: ((URL, URL) -> Void)?`; when a context is set, the finished clip is kept in the store before the Photos save.

- [ ] **Step 1: Write the failing test**

```swift
// app/FollowCamTests/SessionStoreTests.swift
import XCTest
@testable import FollowCam

final class SessionStoreTests: XCTestCase {
    func testKeepCopiesClipAndWritesSidecarNextToIt() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let store = SessionStore(root: tmp)
        let clip = tmp.appendingPathComponent("in.mp4")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        try Data("video".utf8).write(to: clip)
        let ctx = RecordingContext(session: "s1", view: "v1", label: "left", hostOffsetMs: 5, hostOffsetErrMs: 1,
                                   recordStartHostMs: 123)
        let kept = try store.keep(clip: clip, context: ctx, fps: 30, width: 1080, height: 1920)
        XCTAssertEqual(kept.clip.lastPathComponent, "v1.mp4")
        XCTAssertEqual(kept.sidecar.lastPathComponent, "v1.followcam.json")
        let side = try JSONDecoder().decode(ViewSidecar.self, from: Data(contentsOf: kept.sidecar))
        XCTAssertEqual(side.recordStartHostMs, 123)
        XCTAssertEqual(side.orientation, "portrait")
        XCTAssertEqual(store.sessions().map(\.id), ["s1"])
        XCTAssertEqual(store.sessions()[0].clips.count, 1)
        store.remove(session: "s1")
        XCTAssertTrue(store.sessions().isEmpty)
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd app && xcodegen generate && xcodebuild test -scheme FollowCam -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | grep -E "error:|failed" | head -5`
Expected: build error `cannot find 'SessionStore' in scope`.

- [ ] **Step 3: Implement `SessionStore.swift`**

```swift
// app/FollowCamApp/Session/SessionStore.swift
import Foundation
import UIKit

struct RecordingContext {
    let session: String
    let view: String
    let label: String
    let hostOffsetMs: Double
    let hostOffsetErrMs: Double
    var recordStartHostMs: Double
}

/// Documents/Sessions/<session>/<view>.mp4 + <view>.followcam.json (+ session.json on the host).
/// Clips are kept here until uploaded; Photos gets its own copy as before.
final class SessionStore {
    struct StoredSession: Identifiable {
        let id: String
        let clips: [(clip: URL, sidecar: URL)]
        let manifest: URL?
    }

    static let defaultRoot = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Sessions")
    let root: URL

    init(root: URL = SessionStore.defaultRoot) { self.root = root }

    func keep(clip: URL, context: RecordingContext, fps: Int, width: Int, height: Int) throws -> (clip: URL, sidecar: URL) {
        let dir = root.appendingPathComponent(context.session)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let dst = dir.appendingPathComponent("\(context.view).mp4")
        if FileManager.default.fileExists(atPath: dst.path) { try FileManager.default.removeItem(at: dst) }
        try FileManager.default.copyItem(at: clip, to: dst)
        let sidecar = ViewSidecar(session: context.session, view: context.view, device: Self.deviceModel(),
                                  hostOffsetMs: context.hostOffsetMs, hostOffsetErrMs: context.hostOffsetErrMs,
                                  recordStartHostMs: context.recordStartHostMs, fps: fps, width: width, height: height,
                                  orientation: width >= height ? "landscape" : "portrait", label: context.label)
        let side = dir.appendingPathComponent("\(context.view).followcam.json")
        try JSONEncoder().encode(sidecar).write(to: side)
        return (dst, side)
    }

    func writeManifest(_ m: SessionManifest) throws -> URL {
        let dir = root.appendingPathComponent(m.session)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("session.json")
        try JSONEncoder().encode(m).write(to: url)
        return url
    }

    func sessions() -> [StoredSession] {
        let fm = FileManager.default
        guard let dirs = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) else { return [] }
        return dirs.filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .map { dir in
                let files = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
                let clips = files.filter { $0.pathExtension == "mp4" }.compactMap { clip -> (URL, URL)? in
                    let side = dir.appendingPathComponent(clip.deletingPathExtension().lastPathComponent + ".followcam.json")
                    return fm.fileExists(atPath: side.path) ? (clip, side) : nil
                }
                let manifest = dir.appendingPathComponent("session.json")
                return StoredSession(id: dir.lastPathComponent, clips: clips,
                                     manifest: fm.fileExists(atPath: manifest.path) ? manifest : nil)
            }
    }

    func remove(session: String) {
        try? FileManager.default.removeItem(at: root.appendingPathComponent(session))
    }

    static func deviceModel() -> String {
        var sys = utsname(); uname(&sys)
        return withUnsafePointer(to: &sys.machine) { $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) } }
    }
}
```

- [ ] **Step 4: Wire the recorder**

In `CameraManager.swift`:

Add properties after `private var sessionStarted = false` (line 41 area):

```swift
    /// Set by the Sessions screen while a session is active; nil = plain recording.
    var recordingContext: RecordingContext?
    /// Called on the capture queue once the clip and sidecar are in the SessionStore.
    var onKept: ((URL, URL) -> Void)?
    private let store = SessionStore()
```

In `stopRecording()` replace the last line `writer.finishWriting { self.saveToPhotos(url) }` with:

```swift
            let context = self.recordingContext
            writer.finishWriting {
                if let context {
                    do {
                        let kept = try self.store.keep(clip: url, context: context, fps: 30, width: 1080, height: 1920)
                        self.onKept?(kept.clip, kept.sidecar)
                    } catch {
                        DispatchQueue.main.async { self.saveState = .failed("Could not keep the clip for upload: \(error.localizedDescription)") }
                    }
                }
                self.saveToPhotos(url)
            }
```

`recordStartHostMs` is filled by the caller (Task 5) right before `startRecording()`, from `SessionCoordinator.recordStartHostMs`; the recorder itself does not know host time.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd app && xcodegen generate && xcodebuild test -scheme FollowCam -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | grep -E "Test Suite|passed|failed" | tail -5`
Expected: all suites pass including `SessionStoreTests`.

- [ ] **Step 6: Commit**

```bash
git add app/FollowCamApp/Session/SessionStore.swift app/FollowCamTests/SessionStoreTests.swift app/FollowCamApp/CameraManager.swift app/FollowCam.xcodeproj/project.pbxproj
git commit -m "app: keep session clips with sidecars in Documents before the Photos save

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

---

### Task 4: Ingest service (FastAPI on the Spark)

**Files:**
- Create: `ingest/__init__.py`, `ingest/app.py`, `ingest/storage.py`, `ingest/watcher.py`, `ingest/teams.example.yaml`, `ingest/requirements.txt`, `ingest/tests/__init__.py`, `ingest/tests/test_api.py`
- Modify: `.gitignore` (add `ingest/teams.yaml`, `/data/sessions/` is already under `/data/`)

**Interfaces:**
- Produces HTTP API (bearer token in `Authorization`):
  - `PUT /sessions/{sid}/manifest` body `session.json` → 204
  - `PUT /sessions/{sid}/views/{view}/sidecar` body sidecar JSON → 204
  - `PUT /sessions/{sid}/views/{view}/clip` raw mp4 body (streamed) → 204
  - `GET /sessions/{sid}` → `{"session", "expected": [...], "arrived": [...], "complete": bool, "pipeline": "idle|running|done|failed", "report": "<url or null>"}`
  - `POST /sessions/{sid}/run` → 202, forces the pipeline on what has arrived
- `storage.Storage(root: Path)` with `put_manifest(sid, data: bytes)`, `put_sidecar(sid, view, data)`, `put_clip(sid, view, stream)`, `status(sid) -> dict`, `path(sid) -> Path`. Files: `data/sessions/<sid>/session.json`, `<view>.followcam.json`, `<view>.mp4`, `pipeline.json` (`{"state": ..., "started": ..., "report": ...}`).
- `watcher.maybe_run(storage, sid, force=False) -> bool` starts `ml/.venv/bin/python -m vision.multiview.run --session <sid>` as a detached subprocess when complete (or forced) and no run is active.

- [ ] **Step 1: Requirements, example teams file, gitignore**

```bash
mkdir -p ingest/tests && touch ingest/__init__.py ingest/tests/__init__.py
cat > ingest/requirements.txt <<'EOF'
fastapi>=0.115
uvicorn>=0.30
httpx>=0.27
python-multipart>=0.0.9
pyyaml>=6
EOF
ml/.venv/bin/python -m pip install -r ingest/requirements.txt
cat > ingest/teams.example.yaml <<'EOF'
# Copy to ingest/teams.yaml (gitignored). One token per team; hand it out in person.
teams:
  - name: "BC Example"
    token: "replace-with-a-long-random-string"
EOF
printf '\n# ingest tokens\ningest/teams.yaml\n' >> .gitignore
```

- [ ] **Step 2: Write the failing tests**

```python
# ingest/tests/test_api.py
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ingest.app import create_app

MANIFEST = {"schema": 1, "session": "s1", "team": "BC Example", "date": "2026-09-16T18:00:00Z",
            "primary": "v1", "views": [{"view": "v1", "label": "left"}, {"view": "v2", "label": "right"}]}
SIDECAR = {"schema": 1, "session": "s1", "view": "v1", "device": "iPhone16,1", "host_offset_ms": 0.0,
           "host_offset_err_ms": 1.0, "record_start_host_ms": 1.0, "fps": 30, "width": 1080, "height": 1920,
           "orientation": "portrait", "label": "left"}


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    teams = tmp_path / "teams.yaml"
    teams.write_text("teams:\n  - name: BC Example\n    token: tok123\n")
    launched = []
    monkeypatch.setattr("ingest.watcher.launch", lambda sid, root: launched.append(sid))
    app = create_app(root=tmp_path / "sessions", teams_file=teams)
    c = TestClient(app)
    c.launched = launched
    return c


AUTH = {"Authorization": "Bearer tok123"}


def test_rejects_bad_token(client):
    r = client.put("/sessions/s1/manifest", json=MANIFEST, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_upload_flow_marks_complete_and_launches(client):
    assert client.put("/sessions/s1/manifest", json=MANIFEST, headers=AUTH).status_code == 204
    assert client.put("/sessions/s1/views/v1/sidecar", json=SIDECAR, headers=AUTH).status_code == 204
    assert client.put("/sessions/s1/views/v1/clip", content=b"mp4bytes", headers=AUTH).status_code == 204
    s = client.get("/sessions/s1", headers=AUTH).json()
    assert s["arrived"] == ["v1"] and s["complete"] is False and client.launched == []
    side2 = {**SIDECAR, "view": "v2", "label": "right"}
    client.put("/sessions/s1/views/v2/sidecar", json=side2, headers=AUTH)
    client.put("/sessions/s1/views/v2/clip", content=b"mp4bytes", headers=AUTH)
    s = client.get("/sessions/s1", headers=AUTH).json()
    assert s["complete"] is True and s["pipeline"] == "running" and client.launched == ["s1"]


def test_clip_lands_on_disk_under_session(client, tmp_path: Path):
    client.put("/sessions/s2/views/vX/clip", content=b"abc", headers=AUTH)
    assert (tmp_path / "sessions" / "s2" / "vX.mp4").read_bytes() == b"abc"


def test_force_run_without_manifest(client):
    client.put("/sessions/s3/views/v1/clip", content=b"abc", headers=AUTH)
    assert client.post("/sessions/s3/run", headers=AUTH).status_code == 202
    assert client.launched == ["s3"]


def test_unknown_session_404(client):
    assert client.get("/sessions/nope", headers=AUTH).status_code == 404
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `ml/.venv/bin/python -m pytest ingest/tests -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest.app'`

- [ ] **Step 4: Implement `storage.py`, `watcher.py`, `app.py`**

```python
# ingest/storage.py
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import BinaryIO

_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def check_id(s: str) -> str:
    if not _ID.match(s):
        raise ValueError(f"bad id {s!r}")
    return s


class Storage:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def path(self, sid: str) -> Path:
        return self.root / check_id(sid)

    def put_manifest(self, sid: str, data: bytes) -> None:
        json.loads(data)  # must be JSON
        d = self.path(sid); d.mkdir(exist_ok=True)
        (d / "session.json").write_bytes(data)

    def put_sidecar(self, sid: str, view: str, data: bytes) -> None:
        json.loads(data)
        d = self.path(sid); d.mkdir(exist_ok=True)
        (d / f"{check_id(view)}.followcam.json").write_bytes(data)

    def put_clip(self, sid: str, view: str, stream: BinaryIO) -> int:
        d = self.path(sid); d.mkdir(exist_ok=True)
        tmp = d / f"{check_id(view)}.mp4.part"
        with tmp.open("wb") as f:
            shutil.copyfileobj(stream, f, 1 << 20)
        tmp.replace(d / f"{view}.mp4")
        return (d / f"{view}.mp4").stat().st_size

    def exists(self, sid: str) -> bool:
        return self.path(sid).is_dir()

    def status(self, sid: str) -> dict:
        d = self.path(sid)
        manifest = json.loads((d / "session.json").read_text()) if (d / "session.json").exists() else None
        expected = [v["view"] for v in manifest["views"]] if manifest else []
        arrived = sorted(p.stem for p in d.glob("*.mp4") if (d / f"{p.stem}.followcam.json").exists())
        pipe = json.loads((d / "pipeline.json").read_text()) if (d / "pipeline.json").exists() else {"state": "idle", "report": None}
        return {"session": sid, "expected": expected, "arrived": arrived,
                "complete": bool(expected) and set(expected) <= set(arrived),
                "pipeline": pipe.get("state", "idle"), "report": pipe.get("report")}

    def set_pipeline(self, sid: str, state: str, report: str | None = None) -> None:
        (self.path(sid) / "pipeline.json").write_text(json.dumps({"state": state, "report": report}))
```

```python
# ingest/watcher.py
"""Start the multi-view pipeline for a session when every expected view has arrived."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ingest.storage import Storage

REPO = Path(__file__).resolve().parents[1]


def launch(sid: str, root: Path) -> None:
    """Detached subprocess; the pipeline writes pipeline.json itself when done."""
    log = root / sid / "pipeline.log"
    subprocess.Popen([sys.executable, "-m", "vision.multiview.run", "--session", sid, "--sessions-root", str(root)],
                     cwd=REPO, stdout=log.open("ab"), stderr=subprocess.STDOUT, start_new_session=True)


def maybe_run(storage: Storage, sid: str, force: bool = False) -> bool:
    st = storage.status(sid)
    if st["pipeline"] == "running":
        return False
    if not (force or st["complete"]):
        return False
    storage.set_pipeline(sid, "running")
    launch(sid, storage.root)
    return True
```

```python
# ingest/app.py
"""FollowCam ingest: clips + sidecars in, sessions on disk, pipeline kicked off.

  ml/.venv/bin/python -m uvicorn ingest.app:app --host 0.0.0.0 --port 8700
"""
from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import Depends, FastAPI, HTTPException, Request, Response

from ingest import watcher
from ingest.storage import Storage

REPO = Path(__file__).resolve().parents[1]


def create_app(root: Path = REPO / "data" / "sessions", teams_file: Path = REPO / "ingest" / "teams.yaml") -> FastAPI:
    app = FastAPI(title="FollowCam ingest")
    storage = Storage(root)
    tokens = {t["token"]: t["name"] for t in yaml.safe_load(teams_file.read_text())["teams"]} if teams_file.exists() else {}

    def team(request: Request) -> str:
        auth = request.headers.get("Authorization", "")
        tok = auth.removeprefix("Bearer ").strip()
        if tok not in tokens:
            raise HTTPException(401, "bad token")
        return tokens[tok]

    def guard(f):
        try:
            return f()
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.put("/sessions/{sid}/manifest", status_code=204)
    async def put_manifest(sid: str, request: Request, _: str = Depends(team)):
        body = await request.body()
        guard(lambda: storage.put_manifest(sid, body))
        watcher.maybe_run(storage, sid)
        return Response(status_code=204)

    @app.put("/sessions/{sid}/views/{view}/sidecar", status_code=204)
    async def put_sidecar(sid: str, view: str, request: Request, _: str = Depends(team)):
        body = await request.body()
        guard(lambda: storage.put_sidecar(sid, view, body))
        watcher.maybe_run(storage, sid)
        return Response(status_code=204)

    @app.put("/sessions/{sid}/views/{view}/clip", status_code=204)
    async def put_clip(sid: str, view: str, request: Request, _: str = Depends(team)):
        import io
        # request.stream() is async; collect to a temp file in chunks to keep memory flat
        d = guard(lambda: storage.path(sid)); d.mkdir(exist_ok=True)
        tmp = d / f"{view}.upload"
        with tmp.open("wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
        with tmp.open("rb") as f:
            guard(lambda: storage.put_clip(sid, view, f))
        tmp.unlink(missing_ok=True)
        watcher.maybe_run(storage, sid)
        return Response(status_code=204)

    @app.get("/sessions/{sid}")
    def get_session(sid: str, _: str = Depends(team)):
        if not guard(lambda: storage.exists(sid)):
            raise HTTPException(404, "no such session")
        return storage.status(sid)

    @app.post("/sessions/{sid}/run", status_code=202)
    def run(sid: str, _: str = Depends(team)):
        if not guard(lambda: storage.exists(sid)):
            raise HTTPException(404, "no such session")
        watcher.maybe_run(storage, sid, force=True)
        return {"ok": True}

    return app


app = create_app()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `ml/.venv/bin/python -m pytest ingest/tests -q`
Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add .gitignore ingest
git commit -m "ingest: FastAPI upload service with per-team tokens and pipeline trigger

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

---

### Task 5: Uploader and Sessions screen (iOS)

**Files:**
- Create: `app/FollowCamApp/Session/Uploader.swift`, `app/FollowCamApp/Session/SessionsView.swift`
- Modify: `app/FollowCamApp/ContentView.swift` (status strip button near line 210; record button action near line 238)

**Interfaces:**
- Consumes: `SessionStore`, `SessionCoordinator`, `CameraManager.recordingContext/onKept`, `Keychain.get/set` (exists in the app for WHOOP keys; same API), ingest API from Task 4.
- Produces: `final class Uploader: NSObject, ObservableObject, URLSessionTaskDelegate` with `@Published var state: [String: FileState]` keyed by file path (`.queued, .uploading(fraction), .done, .failed(String)`), `func configure(baseURL: URL, token: String)`, `func upload(session: SessionStore.StoredSession)` (manifest if present, then per clip: sidecar then clip), background `URLSession` identifier `com.followcam.upload`. Settings keys: `ingest.url` (UserDefaults), `ingest.token` (Keychain key `KeychainKey.ingestToken`).

- [ ] **Step 1: Implement `Uploader.swift`**

```swift
// app/FollowCamApp/Session/Uploader.swift
import Foundation

/// Background uploads to the ingest service. Sidecar and manifest are small
/// and go as data tasks; clips go as background upload tasks from file so
/// they continue when the app is suspended. A failed clip is re-queued from
/// the start on the next `upload` call (no byte-range resume).
final class Uploader: NSObject, ObservableObject, URLSessionTaskDelegate, URLSessionDataDelegate {
    enum FileState: Equatable { case queued, uploading(Double), done, failed(String) }

    @Published var state: [String: FileState] = [:]
    private var baseURL: URL?
    private var token = ""
    private var taskFiles: [Int: String] = [:]
    private lazy var background: URLSession = {
        let cfg = URLSessionConfiguration.background(withIdentifier: "com.followcam.upload")
        cfg.isDiscretionary = false
        cfg.sessionSendsLaunchEvents = true
        return URLSession(configuration: cfg, delegate: self, delegateQueue: nil)
    }()

    func configure(baseURL: URL, token: String) {
        self.baseURL = baseURL
        self.token = token
    }

    func upload(session: SessionStore.StoredSession) {
        guard let baseURL else { return }
        if let manifest = session.manifest {
            put(url: baseURL.appendingPathComponent("sessions/\(session.id)/manifest"), file: manifest, background: false)
        }
        for (clip, sidecar) in session.clips {
            let view = clip.deletingPathExtension().lastPathComponent
            put(url: baseURL.appendingPathComponent("sessions/\(session.id)/views/\(view)/sidecar"), file: sidecar, background: false)
            put(url: baseURL.appendingPathComponent("sessions/\(session.id)/views/\(view)/clip"), file: clip, background: true)
        }
    }

    private func put(url: URL, file: URL, background bg: Bool) {
        if case .done = state[file.path] { return }
        if case .uploading = state[file.path] { return }
        var req = URLRequest(url: url)
        req.httpMethod = "PUT"
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.setValue(bg ? "video/mp4" : "application/json", forHTTPHeaderField: "Content-Type")
        let task = bg ? background.uploadTask(with: req, fromFile: file)
                      : URLSession.shared.uploadTask(with: req, fromFile: file) { [weak self] _, resp, err in
                          self?.finish(path: file.path, response: resp, error: err)
                      }
        taskFiles[task.taskIdentifier] = file.path
        DispatchQueue.main.async { self.state[file.path] = .queued }
        task.resume()
    }

    private func finish(path: String, response: URLResponse?, error: Error?) {
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        DispatchQueue.main.async {
            if let error { self.state[path] = .failed(error.localizedDescription) }
            else if (200..<300).contains(code) { self.state[path] = .done }
            else { self.state[path] = .failed("HTTP \(code)") }
        }
    }

    // background session delegate
    func urlSession(_ session: URLSession, task: URLSessionTask, didSendBodyData bytesSent: Int64,
                    totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
        guard let path = taskFiles[task.taskIdentifier], totalBytesExpectedToSend > 0 else { return }
        let f = Double(totalBytesSent) / Double(totalBytesExpectedToSend)
        DispatchQueue.main.async { self.state[path] = .uploading(f) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let path = taskFiles.removeValue(forKey: task.taskIdentifier) else { return }
        finish(path: path, response: task.response, error: error)
    }
}
```

- [ ] **Step 2: Implement `SessionsView.swift`**

```swift
// app/FollowCamApp/Session/SessionsView.swift
import SwiftUI

/// Create or join a session, see who is in it and their clock offsets, and
/// watch uploads. The record button on the main screen drives the session
/// while one is active.
struct SessionsView: View {
    @ObservedObject var coordinator: SessionCoordinator
    @ObservedObject var uploader: Uploader
    let store: SessionStore
    @State private var team = UserDefaults.standard.string(forKey: "session.team") ?? ""
    @State private var ingestURL = UserDefaults.standard.string(forKey: "ingest.url") ?? ""
    @State private var ingestToken = Keychain.get(KeychainKey.ingestToken) ?? ""
    @State private var stored: [SessionStore.StoredSession] = []

    var body: some View {
        NavigationStack {
            List {
                Section("This phone") {
                    TextField("Camera label (e.g. baseline left)", text: $coordinator.label)
                }
                Section("Session") {
                    switch coordinator.role {
                    case .none:
                        TextField("Team", text: $team)
                        Button("Host a new session") {
                            UserDefaults.standard.set(team, forKey: "session.team")
                            coordinator.host(team: team)
                        }.disabled(team.isEmpty)
                        Button("Join a session nearby") { coordinator.browse() }
                    case .host:
                        LabeledContent("Code", value: coordinator.code).font(.system(.title2, design: .monospaced))
                        ForEach(coordinator.members) { m in
                            LabeledContent(m.label, value: m.offsetMs.map { String(format: "%+.0f ms ±%.0f", $0, m.errMs ?? 0) } ?? "syncing…")
                        }
                        Button("Leave", role: .destructive) { coordinator.leave() }
                    case .joiner:
                        if coordinator.offset == nil {
                            ForEach(coordinator.nearby) { h in
                                Button("\(h.team)  \(h.code)") { coordinator.join(h) }
                            }
                            if coordinator.nearby.isEmpty { Text("Looking for a host…").foregroundStyle(.secondary) }
                        } else {
                            LabeledContent("Joined", value: "\(coordinator.team) \(coordinator.code)")
                            LabeledContent("Clock offset", value: String(format: "%+.0f ms ±%.0f", coordinator.offset!.offsetMs, coordinator.offset!.errMs))
                        }
                        Button("Leave", role: .destructive) { coordinator.leave() }
                    }
                }
                Section("Ingest") {
                    TextField("https://spark.tailnet-name.ts.net:8700", text: $ingestURL)
                        .textInputAutocapitalization(.never).keyboardType(.URL)
                    SecureField("Team token", text: $ingestToken)
                    Button("Save") {
                        UserDefaults.standard.set(ingestURL, forKey: "ingest.url")
                        Keychain.set(ingestToken, for: KeychainKey.ingestToken)
                        configureUploader()
                    }
                }
                Section("Recorded sessions") {
                    ForEach(stored) { s in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(s.id.prefix(8)).font(.system(.body, design: .monospaced))
                            ForEach(s.clips, id: \.clip) { c in
                                HStack {
                                    Text(c.clip.lastPathComponent).font(.caption)
                                    Spacer()
                                    Text(describe(uploader.state[c.clip.path])).font(.caption).foregroundStyle(.secondary)
                                }
                            }
                            HStack {
                                Button("Upload") { configureUploader(); uploader.upload(session: s) }
                                    .disabled(ingestURL.isEmpty || ingestToken.isEmpty)
                                Spacer()
                                Button("Delete", role: .destructive) { store.remove(session: s.id); stored = store.sessions() }
                            }
                        }
                    }
                    if stored.isEmpty { Text("Nothing recorded yet").foregroundStyle(.secondary) }
                }
            }
            .navigationTitle("Sessions")
            .onAppear { stored = store.sessions(); configureUploader() }
        }
    }

    private func configureUploader() {
        guard let url = URL(string: ingestURL), !ingestToken.isEmpty else { return }
        uploader.configure(baseURL: url, token: ingestToken)
    }

    private func describe(_ s: Uploader.FileState?) -> String {
        switch s {
        case .none: return "not uploaded"
        case .queued: return "queued"
        case .uploading(let f): return String(format: "%.0f%%", f * 100)
        case .done: return "uploaded"
        case .failed(let why): return "failed: \(why)"
        }
    }
}
```

Add `static let ingestToken = "ingest.token"` to the existing `KeychainKey` enum in `app/FollowCamApp/WhoopSync.swift:55`. The `Keychain.set(_ value:, for key:)` and `Keychain.get(_ key:)` helpers already exist in that file.

- [ ] **Step 3: Wire `ContentView`**

Add state objects next to the existing ones (line 15 area):

```swift
    @StateObject private var sessions = SessionCoordinator()
    @StateObject private var uploader = Uploader()
    @State private var showSessions = false
    private let store = SessionStore()
```

Add a Sessions button beside the settings button in `statusStrip` (line 210 area):

```swift
            Button { showSessions = true } label: {
                Image(systemName: sessions.role == .none ? "person.2" : "person.2.fill")
                    .foregroundColor(sessions.role == .none ? .white : .green)
            }
```

Add the sheet after the existing `.sheet(isPresented: $showSettings)`:

```swift
        .sheet(isPresented: $showSessions) { SessionsView(coordinator: sessions, uploader: uploader, store: store) }
```

In `.onAppear`, hook the session to the recorder:

```swift
            sessions.onRecord = { startHostMs in
                camera.recordingContext = RecordingContext(
                    session: sessions.sessionId ?? "solo", view: sessions.viewId, label: sessions.label,
                    hostOffsetMs: sessions.offset?.offsetMs ?? 0, hostOffsetErrMs: sessions.offset?.errMs ?? 0,
                    recordStartHostMs: startHostMs)
                let delay = max(0, (startHostMs - sessions.hostNowMs()) / 1000)
                DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                    if !camera.isRecording { camera.startRecording(); recordingStarted = Date() }
                }
            }
            sessions.onStop = { if camera.isRecording { camera.stopRecording() } }
            camera.onKept = { _, _ in
                if sessions.role == .host, let sid = sessions.sessionId {
                    let m = SessionManifest(session: sid, team: sessions.team, date: ISO8601DateFormatter().string(from: Date()),
                                            primary: sessions.viewId,
                                            views: sessions.members.map { .init(view: $0.id, label: $0.label) })
                    _ = try? store.writeManifest(m)
                }
            }
```

Refactor the record button (`ContentView.swift:238-260`) so the bookkeeping lives in two functions and the button only decides who drives. Replace the `RecordButton` closure body with:

```swift
            RecordButton(isRecording: camera.isRecording) {
                UIImpactFeedbackGenerator(style: .rigid).impactOccurred()
                switch sessions.role {
                case .host: camera.isRecording ? sessions.stopRecording() : sessions.startRecording()
                case .joiner: break   // the host starts and stops everyone
                case .none: camera.isRecording ? stopLocalRecording() : startLocalRecording()
                }
            }
```

and add these two methods to `ContentView` (the bodies are the old `if`/`else` branches, unchanged):

```swift
    private func startLocalRecording() {
        recFrames = 0
        recTargetFrames = 0
        recHRRange = nil
        recSwept = pan.angle...pan.angle
        heart.beginLog()
        camera.startRecording()
        recordingStarted = Date()
    }

    private func stopLocalRecording() {
        camera.stopRecording()
        heart.endLog(stamp: Int(Date().timeIntervalSince1970))
        if let started = recordingStarted {
            summary = SessionSummary(
                duration: Date().timeIntervalSince(started),
                coverageDeg: recSwept.upperBound - recSwept.lowerBound,
                targetHeld: recFrames > 0 ? Double(recTargetFrames) / Double(recFrames) : 0,
                hr: recHRRange)
        }
        recordingStarted = nil
    }
```

Then in the `.onAppear` hooks above, replace `if !camera.isRecording { camera.startRecording(); recordingStarted = Date() }` with `if !camera.isRecording { startLocalRecording() }` and `sessions.onStop = { if camera.isRecording { camera.stopRecording() } }` with `sessions.onStop = { if camera.isRecording { stopLocalRecording() } }`, so a session-driven recording gets the same HR log and summary card as a plain one.

- [ ] **Step 4: Build and run the unit tests**

Run: `cd app && xcodegen generate && xcodebuild test -scheme FollowCam -destination 'platform=iOS Simulator,name=iPhone 17 Pro' 2>&1 | grep -E "BUILD|Test Suite|passed|failed" | tail -5`
Expected: `** TEST SUCCEEDED **`, all suites pass.

- [ ] **Step 5: Two-phone clap test (manual, milestone B-record)**

On two iPhones: host on one (team "Test"), join on the other, wait for the offset to show, press record on the host, clap once loudly near both phones, stop after 10 s. In the Sessions screen both phones list the clip. AirDrop or upload both `*.mp4` and `*.followcam.json` to the Mac into `data/sessions/<id>/`. Then run the alignment check from Task 6 step 6 once that task exists; the sidecar offsets should agree with the audio peak within 50 ms. Record the numbers in `docs/MULTIVIEW.md` (Task 9).

- [ ] **Step 6: Commit**

```bash
git add app/FollowCamApp/Session/Uploader.swift app/FollowCamApp/Session/SessionsView.swift app/FollowCamApp/ContentView.swift app/FollowCamApp/*.swift app/FollowCam.xcodeproj/project.pbxproj
git commit -m "app: sessions screen, background uploader, session-driven record button

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

---

### Task 6: Alignment (sidecar, then audio)

**Files:**
- Create: `vision/multiview/__init__.py`, `vision/multiview/align.py`, `vision/multiview/tests/__init__.py`, `vision/multiview/tests/test_align.py`

**Interfaces:**
- Produces: `align.coarse_offsets(sidecars: dict[str, dict], primary: str) -> dict[str, float]` (view → ms to add to a view's local clip time to get primary time; primary maps to 0.0), `align.envelope(samples: np.ndarray, sr: int, hop_ms: int = 5) -> np.ndarray`, `align.refine(env_primary: np.ndarray, env_view: np.ndarray, hop_ms: int, coarse_ms: float, window_ms: float = 500.0) -> tuple[float, float]` (refined offset ms, confidence = peak over median), `align.extract_audio(clip: Path, sr: int = 8000) -> np.ndarray` (mono float32 via ffmpeg), `align.align_session(session_dir: Path) -> dict` writing `out/<session>/align.json` as `{"primary": id, "slot_ms": 100, "views": {id: {"offset_ms", "method": "sidecar"|"audio", "confidence", "flag": null|"audio_disagrees"|"no_audio"}}}`.

Offset convention: `t_primary = t_view + offset_ms[view]`. From the sidecars, `record_start_host_ms` is already in host time on every phone, so `offset_ms[view] = start_host[view] - start_host[primary]`.

- [ ] **Step 1: Write the failing tests**

```python
# vision/multiview/tests/test_align.py
import numpy as np

from vision.multiview.align import coarse_offsets, envelope, refine


def _side(view, start):
    return {"view": view, "record_start_host_ms": start, "host_offset_ms": 0, "host_offset_err_ms": 2}


def test_coarse_offsets_are_relative_to_primary_start():
    sc = {"A": _side("A", 1000.0), "B": _side("B", 1250.0), "C": _side("C", 900.0)}
    assert coarse_offsets(sc, "A") == {"A": 0.0, "B": 250.0, "C": -100.0}


def test_refine_recovers_a_known_shift():
    rng = np.random.default_rng(0)
    sr, hop = 8000, 5
    base = rng.normal(size=sr * 6).astype(np.float32) * 0.01
    base[sr * 2: sr * 2 + 400] += 1.0          # a clap at 2.0 s
    shift = int(0.180 * sr)                     # view starts 180 ms later -> clap at 1.82 s in the view
    view = np.concatenate([base[shift:], np.zeros(shift, np.float32)])
    ep, ev = envelope(base, sr, hop), envelope(view, sr, hop)
    off, conf = refine(ep, ev, hop, coarse_ms=150.0, window_ms=500.0)
    assert abs(off - 180.0) <= hop
    assert conf > 3


def test_refine_stays_inside_window():
    sr, hop = 8000, 5
    a = np.zeros(sr * 3, np.float32); a[sr] = 1.0
    b = np.zeros(sr * 3, np.float32); b[sr + int(0.9 * sr)] = 1.0   # true offset -900 ms, outside ±500
    off, _ = refine(envelope(a, sr, hop), envelope(b, sr, hop), hop, coarse_ms=0.0, window_ms=500.0)
    assert -500 <= off <= 500
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `ml/.venv/bin/python -m pytest vision/multiview/tests/test_align.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vision.multiview'`

- [ ] **Step 3: Implement `align.py`**

```python
# vision/multiview/align.py
"""Align the views of a session to the primary view: sidecar first, audio refine.

  ml/.venv/bin/python -m vision.multiview.align --session <id> [--sessions-root data/sessions] [--out out/<id>]

t_primary_ms = t_view_ms + offset_ms[view]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SLOT_MS = 100
MAX_DISAGREE_MS = 300.0


def load_session(session_dir: Path) -> tuple[dict, dict[str, dict]]:
    manifest = json.loads((session_dir / "session.json").read_text()) if (session_dir / "session.json").exists() else None
    sidecars = {}
    for p in sorted(session_dir.glob("*.followcam.json")):
        s = json.loads(p.read_text())
        if (session_dir / f"{s['view']}.mp4").exists():
            sidecars[s["view"]] = s
    if not sidecars:
        raise SystemExit(f"{session_dir}: no clips with sidecars")
    if manifest is None:
        first = sorted(sidecars)[0]
        manifest = {"session": session_dir.name, "primary": first,
                    "views": [{"view": v, "label": s.get("label", v)} for v, s in sidecars.items()]}
    if manifest["primary"] not in sidecars:
        manifest["primary"] = sorted(sidecars)[0]
    return manifest, sidecars


def coarse_offsets(sidecars: dict[str, dict], primary: str) -> dict[str, float]:
    p0 = float(sidecars[primary]["record_start_host_ms"])
    return {v: float(s["record_start_host_ms"]) - p0 for v, s in sidecars.items()}


def extract_audio(clip: Path, sr: int = 8000) -> np.ndarray:
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", str(clip), "-vn", "-ac", "1",
           "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=False).stdout
    return np.frombuffer(out, dtype=np.float32)


def envelope(samples: np.ndarray, sr: int, hop_ms: int = 5) -> np.ndarray:
    hop = max(1, sr * hop_ms // 1000)
    n = len(samples) // hop
    if n == 0:
        return np.zeros(0, np.float32)
    env = np.abs(samples[: n * hop]).reshape(n, hop).mean(axis=1)
    return (env - env.mean()).astype(np.float32)


def refine(env_primary: np.ndarray, env_view: np.ndarray, hop_ms: int, coarse_ms: float,
           window_ms: float = 500.0) -> tuple[float, float]:
    """Search offsets in [coarse-window, coarse+window]; return (offset_ms, confidence)."""
    lo, hi = int(np.floor((coarse_ms - window_ms) / hop_ms)), int(np.ceil((coarse_ms + window_ms) / hop_ms))
    scores = []
    for k in range(lo, hi + 1):
        # t_primary = t_view + k*hop  ->  primary[i + k] pairs with view[i]
        if k >= 0:
            a, b = env_primary[k:], env_view[: len(env_primary) - k]
        else:
            a, b = env_primary[: len(env_primary) + k], env_view[-k:]
        n = min(len(a), len(b))
        scores.append(float(np.dot(a[:n], b[:n])) if n > 0 else 0.0)
    scores = np.array(scores)
    best = int(np.argmax(scores))
    med = float(np.median(np.abs(scores))) or 1e-9
    return float((lo + best) * hop_ms), float(scores[best] / med)


def align_session(session_dir: Path, out_dir: Path) -> dict:
    manifest, sidecars = load_session(session_dir)
    primary = manifest["primary"]
    coarse = coarse_offsets(sidecars, primary)
    hop_ms = 5
    prim_audio = extract_audio(session_dir / f"{primary}.mp4")
    prim_env = envelope(prim_audio, 8000, hop_ms) if len(prim_audio) else None
    views = {}
    for v in sidecars:
        entry = {"offset_ms": coarse[v], "method": "sidecar", "confidence": None, "flag": None}
        if v != primary and prim_env is not None:
            audio = extract_audio(session_dir / f"{v}.mp4")
            if len(audio) == 0:
                entry["flag"] = "no_audio"
            else:
                off, conf = refine(prim_env, envelope(audio, 8000, hop_ms), hop_ms, coarse[v])
                if abs(off - coarse[v]) > MAX_DISAGREE_MS:
                    entry["flag"] = "audio_disagrees"
                    entry["confidence"] = conf
                else:
                    entry.update(offset_ms=off, method="audio", confidence=conf)
        elif v != primary:
            entry["flag"] = "no_audio"
        views[v] = entry
    result = {"session": manifest["session"], "primary": primary, "slot_ms": SLOT_MS, "views": views}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "align.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--sessions-root", type=Path, default=REPO / "data" / "sessions")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or REPO / "out" / a.session
    print(json.dumps(align_session(a.sessions_root / a.session, out), indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `ml/.venv/bin/python -m pytest vision/multiview/tests/test_align.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add vision/multiview/__init__.py vision/multiview/align.py vision/multiview/tests
git commit -m "multiview: sidecar plus audio alignment of session views

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

- [ ] **Step 6: Check against the two-phone clap recording (once Task 5 step 5 has produced it)**

Run: `ml/.venv/bin/python -m vision.multiview.align --session <id>`
Expected: for the joiner view, `method: audio`, `confidence` above 3, and `|offset_ms - sidecar offset|` under 50 ms. Write the numbers into `docs/MULTIVIEW.md` in Task 9.

---

### Task 7: Per-view detection and stats

**Files:**
- Create: `vision/multiview/detect.py`, `vision/multiview/tests/test_detect.py`

**Interfaces:**
- Consumes: `vision/track/run.py` CLI (`--video --weights --imgsz --out --overlay --no-overlay --device --publish --conf-ball`), `vision/stats/build.py` CLI (`--tracks --clip --out-dir`), optional `out/<session>/<view>/court_calib.json`.
- Produces: `detect.track_command(clip: Path, out_dir: Path, weights: Path, device: str, imgsz: int = 1280) -> list[str]`, `detect.stats_command(clip: Path, out_dir: Path, calib: Path | None) -> list[str]`, `detect.run_views(session_dir: Path, out_dir: Path, views: list[str], weights: Path, device: str, force: bool = False) -> dict[str, dict]` (view → `{"tracks", "events", "stats"}` paths); files `out/<session>/<view>/tracks.jsonl`, `tracks_meta.json`, `events.json`, `stats.json`.

- [ ] **Step 1: Write the failing tests**

```python
# vision/multiview/tests/test_detect.py
from pathlib import Path

from vision.multiview.detect import stats_command, track_command


def test_track_command_uses_one_model_and_publishes():
    cmd = track_command(Path("d/s1/v1.mp4"), Path("out/s1/v1"), Path("models/followcam_s_1280.pt"), "0")
    assert cmd[1:4] == ["-m", "vision.track.run", "--video"]
    assert "--weights" in cmd and "--publish" in cmd and "--no-overlay" in cmd
    assert cmd[cmd.index("--out") + 1] == "out/s1/v1/tracks.jsonl"
    assert cmd[cmd.index("--device") + 1] == "0"


def test_stats_command_adds_calib_only_when_given():
    base = stats_command(Path("d/s1/v1.mp4"), Path("out/s1/v1"), None)
    assert "--calib" not in base and base[base.index("--out-dir") + 1] == "out/s1/v1"
    with_cal = stats_command(Path("d/s1/v1.mp4"), Path("out/s1/v1"), Path("out/s1/v1/court_calib.json"))
    assert with_cal[with_cal.index("--calib") + 1] == "out/s1/v1/court_calib.json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `ml/.venv/bin/python -m pytest vision/multiview/tests/test_detect.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `detect.py`**

```python
# vision/multiview/detect.py
"""Run the single-view tracker and stats on every view of a session, sequentially."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def track_command(clip: Path, out_dir: Path, weights: Path, device: str, imgsz: int = 1280) -> list[str]:
    return [sys.executable, "-m", "vision.track.run", "--video", str(clip), "--weights", str(weights),
            "--imgsz", str(imgsz), "--out", str(out_dir / "tracks.jsonl"), "--overlay", str(out_dir / "overlay.mp4"),
            "--no-overlay", "--publish", "--device", device, "--conf-ball", "0.35",
            "--events", str(out_dir / "events.json"), "--identities", str(out_dir / "identities.json")]


def stats_command(clip: Path, out_dir: Path, calib: Path | None) -> list[str]:
    cmd = [sys.executable, "-m", "vision.stats.build", "--tracks", str(out_dir / "tracks.jsonl"),
           "--clip", str(clip), "--out-dir", str(out_dir), "--identities", str(out_dir / "identities.json")]
    if calib is not None:
        cmd += ["--calib", str(calib)]
    return cmd


def run_views(session_dir: Path, out_dir: Path, views: list[str], weights: Path, device: str,
              force: bool = False) -> dict[str, dict]:
    result = {}
    for v in views:
        vdir = out_dir / v
        vdir.mkdir(parents=True, exist_ok=True)
        clip = session_dir / f"{v}.mp4"
        tracks, events = vdir / "tracks.jsonl", vdir / "events.json"
        if force or not (vdir / "tracks_meta.json").exists():
            subprocess.run(track_command(clip, vdir, weights, device), cwd=REPO, check=True)
        calib = vdir / "court_calib.json"
        if force or not events.exists():
            subprocess.run(stats_command(clip, vdir, calib if calib.exists() else None), cwd=REPO, check=True)
        result[v] = {"tracks": tracks, "events": events, "stats": vdir / "stats.json",
                     "calib": calib if calib.exists() else None}
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `ml/.venv/bin/python -m pytest vision/multiview/tests/test_detect.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add vision/multiview/detect.py vision/multiview/tests/test_detect.py
git commit -m "multiview: per-view tracking and stats runner

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

---

### Task 8: Fusion in time

**Files:**
- Create: `vision/multiview/fuse.py`, `vision/multiview/tests/test_fuse.py`

**Interfaces:**
- Consumes: per-view `tracks.jsonl` lines (`{"frame", "t", "players", "ball": {"bbox","center","conf"} | null, "hoops": [{"bbox"}]}`; a ball with `"predicted": true` is not a detection), per-view `events.json` (`{"fps", "clip", "shots": [{"t","frame","player_id","team","made","shooter_foot","hoop_bbox"}]}`), `align.json` offsets, optional per-view calibration via `vision.court.project.load_calibration`.
- Produces:
  - `fuse.ball_timeline(per_view: dict[str, list[tuple[float, float]]], offsets: dict[str, float], slot_ms: int = 100, conf_min: float = 0.35, switch_margin: float = 0.1, switch_slots: int = 3) -> list[dict]` where `per_view[view]` is `[(t_view_ms, conf)]` per processed frame and each slot is `{"t_ms", "covered", "best_view", "views": {view: {"conf"}}}`.
  - `fuse.coverage(slots: list[dict], slot_ms: int) -> dict` → `{"covered_fraction", "longest_gap_s", "gap_start_ms", "slots"}`.
  - `fuse.merge_events(per_view: dict[str, dict], offsets: dict[str, float], window_s: float = 1.5) -> list[dict]` → each `{"t", "made", "team", "player_id", "views": [ids], "verdict_view": id, "hoop_bbox", "shooter_foot"}` in primary seconds.
  - `fuse.fuse_session(out_dir: Path, align: dict, views: dict[str, dict]) -> dict` writing `fused_ball.json` (`{"slot_ms", "views": {id: {"offset_ms"}}, "slots": [...]}`), `fused_events.json` (`{"shots": [...], "primary": id}`), `coverage.json` (coverage dict plus `"single_view": {id: coverage dict}`), and, when at least two views have a calibration, `fused_court.json` (`[{"t_ms", "x_m", "y_m", "n_views"}]`).

- [ ] **Step 1: Write the failing tests**

```python
# vision/multiview/tests/test_fuse.py
from vision.multiview.fuse import ball_timeline, coverage, merge_events


def _pv(confs, step_ms=100):
    return [(i * step_ms, c) for i, c in enumerate(confs)]


def test_timeline_marks_coverage_from_any_view_and_applies_offsets():
    per_view = {"A": _pv([0.9, 0.0, 0.0, 0.0]), "B": _pv([0.0, 0.0, 0.8, 0.8])}
    slots = ball_timeline(per_view, {"A": 0.0, "B": 0.0})
    assert [s["covered"] for s in slots] == [True, False, True, True]
    # B is 200 ms ahead of A: its frames shift right by two slots
    slots = ball_timeline(per_view, {"A": 0.0, "B": 200.0})
    assert [s["covered"] for s in slots] == [True, False, False, False, True, True]


def test_best_view_switch_needs_margin_for_three_slots():
    per_view = {"A": _pv([0.9, 0.6, 0.6, 0.6, 0.6, 0.6]), "B": _pv([0.0, 0.65, 0.75, 0.75, 0.75, 0.75])}
    best = [s["best_view"] for s in ball_timeline(per_view, {"A": 0.0, "B": 0.0})]
    # slot1: B ahead by 0.05 (no); slots 2,3,4 ahead by 0.15 -> switch lands at slot 4
    assert best == ["A", "A", "A", "A", "B", "B"]


def test_coverage_longest_gap():
    slots = [{"t_ms": i * 100, "covered": c, "best_view": None, "views": {}} for i, c in
             enumerate([True, False, False, False, True, False, True])]
    c = coverage(slots, 100)
    assert c["covered_fraction"] == 3 / 7 and c["longest_gap_s"] == 0.3 and c["gap_start_ms"] == 100


def test_merge_events_dedupes_and_takes_verdict_from_biggest_hoop():
    ev = {"A": {"shots": [{"t": 10.0, "made": False, "team": 0, "player_id": 3, "hoop_bbox": [0, 0, 20, 20], "shooter_foot": [1, 1]}]},
          "B": {"shots": [{"t": 9.4, "made": True, "team": 0, "player_id": 9, "hoop_bbox": [0, 0, 60, 60], "shooter_foot": [2, 2]},
                          {"t": 40.0, "made": True, "team": 1, "player_id": 5, "hoop_bbox": [0, 0, 10, 10], "shooter_foot": [3, 3]}]}}
    merged = merge_events(ev, {"A": 0.0, "B": 0.0})
    assert len(merged) == 2
    first = merged[0]
    assert first["views"] == ["A", "B"] and first["verdict_view"] == "B" and first["made"] is True
    assert first["player_id"] == 3    # primary keeps player attribution when it saw the shot
    assert merged[1]["views"] == ["B"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `ml/.venv/bin/python -m pytest vision/multiview/tests/test_fuse.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `fuse.py`**

```python
# vision/multiview/fuse.py
"""Fuse per-view tracks and events in time. Slots are primary-view milliseconds."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

SLOT_MS = 100
CONF_MIN = 0.35
SWITCH_MARGIN = 0.1
SWITCH_SLOTS = 3
EVENT_WINDOW_S = 1.5


def read_ball_confs(tracks: Path) -> list[tuple[float, float]]:
    out = []
    with tracks.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            b = r.get("ball")
            conf = float(b["conf"]) if b and not b.get("predicted") else 0.0
            out.append((float(r["t"]) * 1000.0, conf))
    return out


def ball_timeline(per_view: dict[str, list[tuple[float, float]]], offsets: dict[str, float],
                  slot_ms: int = SLOT_MS, conf_min: float = CONF_MIN, switch_margin: float = SWITCH_MARGIN,
                  switch_slots: int = SWITCH_SLOTS) -> list[dict]:
    # best conf per (view, slot), keyed in primary time
    per_slot: dict[int, dict[str, float]] = defaultdict(dict)
    for view, frames in per_view.items():
        off = offsets.get(view, 0.0)
        for t_ms, conf in frames:
            k = int(math.floor((t_ms + off) / slot_ms))
            if k >= 0:
                per_slot[k][view] = max(per_slot[k].get(view, 0.0), conf)
    if not per_slot:
        return []
    n = max(per_slot) + 1
    slots, best, streak, challenger = [], None, 0, None
    for k in range(n):
        confs = {v: per_slot[k].get(v, 0.0) for v in per_view}
        seen = {v: c for v, c in confs.items() if c >= conf_min}
        if seen:
            top = max(seen, key=seen.get)
            if best is None or best not in seen:
                best, streak, challenger = top, 0, None
            elif top != best and seen[top] - seen[best] >= switch_margin:
                if top == challenger:
                    streak += 1
                else:
                    challenger, streak = top, 1
                if streak >= switch_slots:
                    best, streak, challenger = top, 0, None
            else:
                streak, challenger = 0, None
        slots.append({"t_ms": k * slot_ms, "covered": bool(seen), "best_view": best if seen else None,
                      "views": {v: {"conf": round(c, 3)} for v, c in confs.items()}})
    return slots


def coverage(slots: list[dict], slot_ms: int = SLOT_MS) -> dict:
    best = run = 0
    best_start = start = 0
    for i, s in enumerate(slots):
        if s["covered"]:
            run = 0
            continue
        if run == 0:
            start = i
        run += 1
        if run > best:
            best, best_start = run, start
    n = len(slots)
    return {"covered_fraction": (sum(s["covered"] for s in slots) / n) if n else 0.0,
            "longest_gap_s": best * slot_ms / 1000.0, "gap_start_ms": best_start * slot_ms if best else 0, "slots": n}


def _hoop_area(shot: dict) -> float:
    b = shot.get("hoop_bbox") or [0, 0, 0, 0]
    return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))


def merge_events(per_view: dict[str, dict], offsets: dict[str, float], window_s: float = EVENT_WINDOW_S,
                 primary: str | None = None) -> list[dict]:
    primary = primary or sorted(per_view)[0]
    shots = []
    for view, ev in per_view.items():
        for s in ev.get("shots", []):
            shots.append({**s, "t": float(s["t"]) + offsets.get(view, 0.0) / 1000.0, "_view": view})
    shots.sort(key=lambda s: s["t"])
    merged: list[dict] = []
    for s in shots:
        if merged and s["t"] - merged[-1]["t_first"] <= window_s and s["_view"] not in merged[-1]["views"]:
            g = merged[-1]
            g["views"].append(s["_view"])
            g["_members"].append(s)
        else:
            merged.append({"t_first": s["t"], "views": [s["_view"]], "_members": [s]})
    out = []
    for g in merged:
        members = g["_members"]
        verdict = max(members, key=_hoop_area)
        attrib = next((m for m in members if m["_view"] == primary), members[0])
        out.append({"t": round(sum(m["t"] for m in members) / len(members), 3), "made": bool(verdict["made"]),
                    "team": attrib.get("team"), "player_id": attrib.get("player_id"),
                    "views": sorted(g["views"]), "verdict_view": verdict["_view"],
                    "hoop_bbox": verdict.get("hoop_bbox"), "shooter_foot": attrib.get("shooter_foot")})
    return out


def court_positions(views: dict[str, dict], offsets: dict[str, float], slot_ms: int = SLOT_MS) -> list[dict]:
    """Average ball floor position in court metres per slot across calibrated views."""
    from vision.court.project import load_calibration

    per_slot: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for v, files in views.items():
        if not files.get("calib"):
            continue
        cal = load_calibration(str(files["calib"]))
        with Path(files["tracks"]).open() as f:
            for line in f:
                r = json.loads(line)
                b = r.get("ball")
                if not b or b.get("predicted") or float(b["conf"]) < CONF_MIN:
                    continue
                pt = cal.project(int(r["frame"]), [[b["bbox"][0] + (b["bbox"][2] - b["bbox"][0]) / 2, b["bbox"][3]]])
                if pt is None or any(math.isnan(c) for c in pt[0]):
                    continue
                k = int((float(r["t"]) * 1000.0 + offsets.get(v, 0.0)) // slot_ms)
                per_slot[k].append((pt[0][0], pt[0][1], float(b["conf"])))
    out = []
    for k in sorted(per_slot):
        pts = per_slot[k]
        w = sum(c for _, _, c in pts)
        out.append({"t_ms": k * slot_ms, "x_m": sum(x * c for x, _, c in pts) / w,
                    "y_m": sum(y * c for _, y, c in pts) / w, "n_views": len(pts)})
    return out


def fuse_session(out_dir: Path, align: dict, views: dict[str, dict]) -> dict:
    offsets = {v: e["offset_ms"] for v, e in align["views"].items()}
    primary = align["primary"]
    per_view = {v: read_ball_confs(Path(f["tracks"])) for v, f in views.items()}
    slots = ball_timeline(per_view, offsets)
    cov = coverage(slots)
    cov["single_view"] = {v: coverage(ball_timeline({v: per_view[v]}, {v: offsets[v]})) for v in per_view}
    events = {v: json.loads(Path(f["events"]).read_text()) for v, f in views.items() if Path(f["events"]).exists()}
    shots = merge_events(events, offsets, primary=primary)
    (out_dir / "fused_ball.json").write_text(json.dumps(
        {"slot_ms": SLOT_MS, "views": {v: {"offset_ms": o} for v, o in offsets.items()}, "slots": slots}))
    (out_dir / "fused_events.json").write_text(json.dumps({"primary": primary, "shots": shots}, indent=1))
    (out_dir / "coverage.json").write_text(json.dumps(cov, indent=1))
    calibrated = [v for v, f in views.items() if f.get("calib")]
    if len(calibrated) >= 2:
        (out_dir / "fused_court.json").write_text(json.dumps(court_positions(views, offsets)))
    return {"coverage": cov, "shots": len(shots), "calibrated_views": calibrated}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `ml/.venv/bin/python -m pytest vision/multiview/tests/test_fuse.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add vision/multiview/fuse.py vision/multiview/tests/test_fuse.py
git commit -m "multiview: ball timeline with hysteresis, coverage, event union

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

---

### Task 9: Session runner, report page, Makefile, docs

**Files:**
- Create: `vision/multiview/report.py`, `vision/multiview/run.py`, `vision/multiview/tests/test_run_fixture.py`, `docs/MULTIVIEW.md`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `align.align_session`, `detect.run_views`, `fuse.fuse_session`, `ingest.storage.Storage.set_pipeline` (to record done/failed for the ingest status).
- Produces: `report.build(out_dir: Path, session: str, align: dict, coverage: dict, shots: list[dict], labels: dict[str, str]) -> Path` writing `out/<session>/multiview.html` (coverage row: fused vs each single view, longest gap; a shots table with the views that saw each shot and the verdict view; alignment table with method and flags); `run.run(session: str, sessions_root: Path, out_root: Path, weights: Path, device: str, force: bool) -> dict`; CLI `python -m vision.multiview.run --session <id> [--sessions-root] [--out-root out] [--weights models/followcam_s_1280.pt] [--device 0] [--force]`.

- [ ] **Step 1: Write the fixture test (no video, no model)**

```python
# vision/multiview/tests/test_run_fixture.py
import json
from pathlib import Path

from vision.multiview.fuse import fuse_session
from vision.multiview.report import build


def _tracks(path: Path, confs, fps=10):
    with path.open("w") as f:
        for i, c in enumerate(confs):
            ball = {"bbox": [0, 0, 10, 10], "center": [5, 5], "conf": c} if c > 0 else None
            f.write(json.dumps({"frame": i, "t": i / fps, "players": [], "ball": ball, "hoops": []}) + "\n")


def test_fixture_session_end_to_end(tmp_path: Path):
    out = tmp_path / "out"
    for v, confs in (("A", [0.9, 0.9, 0, 0, 0, 0, 0.9, 0.9]), ("B", [0, 0, 0.8, 0.8, 0, 0, 0, 0])):
        (out / v).mkdir(parents=True)
        _tracks(out / v / "tracks.jsonl", confs)
        (out / v / "events.json").write_text(json.dumps({"fps": 10, "clip": f"{v}.mp4", "shots": []}))
    align = {"session": "fx", "primary": "A", "slot_ms": 100,
             "views": {"A": {"offset_ms": 0.0, "method": "sidecar", "confidence": None, "flag": None},
                       "B": {"offset_ms": 0.0, "method": "audio", "confidence": 5.0, "flag": None}}}
    views = {v: {"tracks": out / v / "tracks.jsonl", "events": out / v / "events.json", "calib": None} for v in "AB"}
    r = fuse_session(out, align, views)
    cov = json.loads((out / "coverage.json").read_text())
    assert cov["covered_fraction"] == 0.75 and cov["longest_gap_s"] == 0.2
    assert cov["single_view"]["A"]["longest_gap_s"] == 0.4
    page = build(out, "fx", align, cov, json.loads((out / "fused_events.json").read_text())["shots"], {"A": "left", "B": "right"})
    html = page.read_text()
    assert "75" in html and "left" in html and "audio" in html
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `ml/.venv/bin/python -m pytest vision/multiview/tests/test_run_fixture.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vision.multiview.report'`

- [ ] **Step 3: Implement `report.py`**

```python
# vision/multiview/report.py
"""Self-contained multiview.html: coverage, per-shot views, alignment."""
from __future__ import annotations

import html
from pathlib import Path


def _row(cells) -> str:
    return "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>"


def build(out_dir: Path, session: str, align: dict, coverage: dict, shots: list[dict], labels: dict[str, str]) -> Path:
    name = lambda v: labels.get(v, v)
    cov_rows = [_row(["fused (any view)", f"{coverage['covered_fraction'] * 100:.0f} %", f"{coverage['longest_gap_s']:.1f} s"])]
    for v, c in coverage.get("single_view", {}).items():
        cov_rows.append(_row([name(v), f"{c['covered_fraction'] * 100:.0f} %", f"{c['longest_gap_s']:.1f} s"]))
    shot_rows = [_row([f"{s['t']:.1f}", "made" if s["made"] else "miss", s.get("team"), s.get("player_id"),
                       ", ".join(name(v) for v in s["views"]), name(s["verdict_view"])]) for s in shots]
    align_rows = [_row([name(v), f"{e['offset_ms']:+.0f}", e["method"],
                        "" if e.get("confidence") is None else f"{e['confidence']:.1f}", e.get("flag") or ""])
                  for v, e in align["views"].items()]
    page = f"""<!doctype html><meta charset=utf-8><title>FollowCam multi-view {html.escape(session)}</title>
<style>body{{font:14px system-ui;margin:24px;color:#eee;background:#111}}table{{border-collapse:collapse;margin:12px 0 28px}}
td,th{{border:1px solid #444;padding:6px 10px;text-align:left}}h2{{margin-top:28px}}</style>
<h1>Session {html.escape(session)}</h1>
<h2>Ball coverage</h2><table><tr><th>View</th><th>Ball seen</th><th>Longest gap</th></tr>{''.join(cov_rows)}</table>
<h2>Shots ({len(shots)})</h2><table><tr><th>t (s)</th><th>Result</th><th>Team</th><th>Player</th><th>Seen by</th><th>Verdict from</th></tr>
{''.join(shot_rows) or _row(['no shots called'] + [''] * 5)}</table>
<h2>Alignment to {html.escape(name(align['primary']))}</h2><table><tr><th>View</th><th>Offset ms</th><th>Method</th><th>Confidence</th><th>Flag</th></tr>{''.join(align_rows)}</table>
"""
    out = out_dir / "multiview.html"
    out.write_text(page)
    return out
```

- [ ] **Step 4: Implement `run.py`**

```python
# vision/multiview/run.py
"""Whole multi-view pipeline for one session: align, track each view, fuse, report.

  ml/.venv/bin/python -m vision.multiview.run --session <id> [--weights models/followcam_s_1280.pt] [--device 0] [--force]
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from vision.multiview.align import align_session, load_session
from vision.multiview.detect import run_views
from vision.multiview.fuse import fuse_session
from vision.multiview.report import build

REPO = Path(__file__).resolve().parents[2]


def run(session: str, sessions_root: Path, out_root: Path, weights: Path, device: str, force: bool = False) -> dict:
    session_dir, out_dir = sessions_root / session, out_root / session
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, sidecars = load_session(session_dir)
    labels = {v["view"]: v.get("label", v["view"]) for v in manifest["views"]}
    align = align_session(session_dir, out_dir)
    views = run_views(session_dir, out_dir, list(sidecars), weights, device, force)
    summary = fuse_session(out_dir, align, views)
    shots = json.loads((out_dir / "fused_events.json").read_text())["shots"]
    cov = json.loads((out_dir / "coverage.json").read_text())
    page = build(out_dir, session, align, cov, shots, labels)
    return {**summary, "report": str(page)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--sessions-root", type=Path, default=REPO / "data" / "sessions")
    ap.add_argument("--out-root", type=Path, default=REPO / "out")
    ap.add_argument("--weights", type=Path, default=REPO / "models" / "followcam_s_1280.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    from ingest.storage import Storage

    storage = Storage(a.sessions_root)
    try:
        r = run(a.session, a.sessions_root, a.out_root, a.weights, a.device, a.force)
        storage.set_pipeline(a.session, "done", r["report"])
        print(json.dumps(r, indent=1, default=str))
    except Exception:
        traceback.print_exc()
        storage.set_pipeline(a.session, "failed")
        raise


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the fixture test**

Run: `ml/.venv/bin/python -m pytest vision/multiview/tests -q`
Expected: all multiview tests pass (`10 passed` across align, detect, fuse, fixture).

- [ ] **Step 6: Makefile targets and docs**

Append to `Makefile`:

```make

# ---- Multi-view sessions (docs/MULTIVIEW.md, docs/superpowers/plans/2026-09-16-multiview-sync.md)
#   make multiview-test                    unit tests
#   make multiview RUN=<session id>        align + track + fuse + report for one uploaded session
#   make ingest-serve                      upload service on :8700 (run on the Spark)
RUN ?=
.PHONY: multiview-test multiview ingest-serve
multiview-test:
	$(MLPY) -m pytest ingest/tests vision/multiview/tests -q
multiview:
	$(MLPY) -m vision.multiview.run --session $(RUN) --weights $(WEIGHTS) --device $(DEVICE)
ingest-serve:
	$(MLPY) -m uvicorn ingest.app:app --host 0.0.0.0 --port 8700
```

Write `docs/MULTIVIEW.md`:

```markdown
# Multi-view sessions

Several phones, one game, one fused analysis. No rig needed.

## On the court
1. Every phone: FollowCam app, Sessions screen, set a camera label ("baseline left").
2. One phone hosts (enter the team name). It shows a 4-letter code.
3. Other phones: Join a session nearby, tap the host with that code. Wait for
   "Clock offset" to show (a few seconds).
4. Host presses record; every phone starts. Host presses stop; every phone stops.
   Clips go to Photos as before and are kept in the app for upload.

## Upload
Sessions screen, Ingest: URL of the ingest service and the team token
(handed out in person). Upload per session. The service runs on the Spark:
`make ingest-serve MLPY=python` inside the container, exposed with Tailscale
(`tailscale serve --https=8700 localhost:8700` or a funnel). Status:
`GET /sessions/<id>`.

## Pipeline
Runs automatically once every view in the manifest has arrived, or
`make multiview RUN=<id>`. Output in `out/<id>/`: `align.json`, one folder
per view (`tracks.jsonl`, `events.json`, `stats.json`), `fused_ball.json`,
`fused_events.json`, `coverage.json`, `multiview.html`.

The number that matters: `coverage.json` `covered_fraction` and
`longest_gap_s` for the fused timeline against each single view.

## Training
`python -m ml.spark.pseudolabel --clip data/sessions/<id>/<view>.mp4 --weights models/followcam_<x>.pt --fused out/<id>/fused_ball.json --view <view>`
puts frames where another phone saw the ball first in the review queue.

## Clap test results
(fill in after the first two-phone test: sidecar offset, audio offset, difference in ms)
```

- [ ] **Step 7: Commit**

```bash
git add vision/multiview/report.py vision/multiview/run.py vision/multiview/tests/test_run_fixture.py Makefile docs/MULTIVIEW.md
git commit -m "multiview: session runner, report page, make targets, docs

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

- [ ] **Step 8: First real session (milestone B-align-fuse)**

With the two-phone clap clips in `data/sessions/<id>/` and a trained model from the Spark plan (or, before that exists, `yolo11n.pt` with `--weights yolo11n.pt` just to exercise the path; its COCO class ids do not match, so coverage will be meaningless): `make multiview RUN=<id> WEIGHTS=models/followcam_s_1280.pt DEVICE=mps`. Open `out/<id>/multiview.html`. Expected: fused coverage at or above the primary view's, and the alignment table shows `audio` with a confidence above 3 for the joiner. Paste the coverage row and the clap numbers into `docs/MULTIVIEW.md` and commit.

---

## Self-review notes

- Spec B1 flow: Tasks 2 (host/join/offset), 3 and 5 (record with host time, sidecar, upload), 4 (ingest), 9 (auto-run, report). Deviation: 4-letter code plus Multipeer browsing instead of a QR code, stated in Task 2.
- Spec B2 sidecar and manifest: Task 1 models, Task 3 writes them, Task 5 writes the manifest on the host.
- Spec B3 app changes: Tasks 2, 3, 5. Rig link, HR, tap-to-track untouched.
- Spec B4 ingest: Task 4, endpoints as listed; "resumable" is retry-from-start of a failed file, stated in Task 5.
- Spec B5 alignment: Task 6, window 500 ms, reject over 300 ms, flags.
- Spec B6 per-view detection: Task 7, unchanged single-view tools.
- Spec B7 fusion: Task 8 (timeline, hysteresis 0.1 over 3 slots, coverage, event union within 1.5 s with largest-hoop verdict, optional court plane); the dashboard row and view selector are delivered as the standalone `multiview.html` in Task 9 rather than edits to `vision/dashboard/build.py`.
- Spec B8 training tie-in: `fused_ball.json` shape matches what `ml/spark/pseudolabel.py` reads in the Spark plan (`slot_ms`, `views.{id}.offset_ms`, `slots[].views.{id}.conf`).
- Testing section: unit tests in Tasks 1, 3, 4, 6, 7, 8, fixture end to end in Task 9, two-phone clap in Task 5.
