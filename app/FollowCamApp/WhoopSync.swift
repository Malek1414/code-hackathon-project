import AuthenticationServices
import Foundation
import Security

// WHOOP REST sync — ported from Malek1414/HUSTLR (Vitality/Whoop*) and
// iterated for FollowCam: adds the v2 workout endpoint with per-zone
// durations + max HR, which calibrates ml/correlate_hr.py far better than
// a guessed max heart rate. Live in-game HR still comes from
// HeartRateMonitor (BLE broadcast); this is the post-game enrichment.
//
// Setup (once, Sammy): register a WHOOP app at developer.whoop.com with
// redirect URI "followcam://whoop-callback", then call
// WhoopClient.configure(clientID:secret:) on first launch.

enum Keychain {
    private static let service = "com.followcam.app"

    static func set(_ value: String, for key: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
        ]
        SecItemDelete(query as CFDictionary)
        var attributes = query
        attributes[kSecValueData as String] = Data(value.utf8)
        attributes[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        SecItemAdd(attributes as CFDictionary, nil)
    }

    static func get(_ key: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func delete(_ key: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
        ]
        SecItemDelete(query as CFDictionary)
    }
}

enum KeychainKey {
    static let whoopClientID = "whoop.clientID"
    static let whoopClientSecret = "whoop.clientSecret"
    static let whoopAccessToken = "whoop.accessToken"
    static let whoopRefreshToken = "whoop.refreshToken"
    static let whoopTokenExpiry = "whoop.tokenExpiry"
}

enum WhoopError: Error, LocalizedError {
    case notConnected
    case authFailed(String)
    case httpError(Int)
    case decodeFailed(String)

    var errorDescription: String? {
        switch self {
        case .notConnected: return "WHOOP is not connected."
        case .authFailed(let reason): return "WHOOP auth failed: \(reason)"
        case .httpError(let code): return "WHOOP HTTP \(code)"
        case .decodeFailed(let detail): return "WHOOP decode failed: \(detail)"
        }
    }
}

struct WhoopPage<T: Decodable & Sendable>: Decodable, Sendable {
    let records: [T]
    let nextToken: String?

    enum CodingKeys: String, CodingKey {
        case records
        case nextToken = "next_token"
    }
}

/// v2/activity/workout — zone durations are the correlation gold.
struct WhoopWorkout: Codable, Sendable {
    struct ZoneDuration: Codable, Sendable {
        let zoneZeroMilli: Int?
        let zoneOneMilli: Int?
        let zoneTwoMilli: Int?
        let zoneThreeMilli: Int?
        let zoneFourMilli: Int?
        let zoneFiveMilli: Int?

        enum CodingKeys: String, CodingKey {
            case zoneZeroMilli = "zone_zero_milli"
            case zoneOneMilli = "zone_one_milli"
            case zoneTwoMilli = "zone_two_milli"
            case zoneThreeMilli = "zone_three_milli"
            case zoneFourMilli = "zone_four_milli"
            case zoneFiveMilli = "zone_five_milli"
        }
    }

    struct Score: Codable, Sendable {
        let strain: Double?
        let averageHeartRate: Double?
        let maxHeartRate: Double?
        let kilojoule: Double?
        let zoneDuration: ZoneDuration?

        enum CodingKeys: String, CodingKey {
            case strain, kilojoule
            case averageHeartRate = "average_heart_rate"
            case maxHeartRate = "max_heart_rate"
            case zoneDuration = "zone_duration"
        }
    }

    let id: String?
    let start: Date?
    let end: Date?
    let sportId: Int?
    let scoreState: String?
    let score: Score?

    enum CodingKeys: String, CodingKey {
        case id, start, end, score
        case sportId = "sport_id"
        case scoreState = "score_state"
    }
}

actor WhoopClient {
    static let shared = WhoopClient()

    static let redirectURI = "followcam://whoop-callback"
    static let scopes = "read:workout read:recovery read:cycles read:profile"
    private static let baseURL = URL(string: "https://api.prod.whoop.com/developer")!
    private static let tokenURL = URL(string: "https://api.prod.whoop.com/oauth/oauth2/token")!
    static let authURL = URL(string: "https://api.prod.whoop.com/oauth/oauth2/auth")!

    static var clientID: String { Keychain.get(KeychainKey.whoopClientID) ?? "" }

    static func configure(clientID: String, secret: String) {
        Keychain.set(clientID, for: KeychainKey.whoopClientID)
        Keychain.set(secret, for: KeychainKey.whoopClientSecret)
    }

    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let string = try container.decode(String.self)
            let iso = ISO8601DateFormatter()
            iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            if let date = iso.date(from: string) { return date }
            iso.formatOptions = [.withInternetDateTime]
            if let date = iso.date(from: string) { return date }
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unparseable date: \(string)")
        }
        return d
    }()

    // Coalesces concurrent token refreshes: WHOOP rotates refresh tokens, so
    // two racing refreshes invalidate each other (lesson learned in HUSTLR).
    private var refreshTask: Task<Void, Error>?

    var isConnected: Bool {
        Keychain.get(KeychainKey.whoopRefreshToken) != nil
    }

    // MARK: OAuth

    func exchangeCode(_ code: String) async throws {
        guard let secret = Keychain.get(KeychainKey.whoopClientSecret) else {
            throw WhoopError.authFailed("missing client secret — call WhoopClient.configure first")
        }
        try await requestToken(form: [
            "grant_type": "authorization_code",
            "code": code,
            "client_id": Self.clientID,
            "client_secret": secret,
            "redirect_uri": Self.redirectURI,
        ])
    }

    private func hasValidAccessToken() -> Bool {
        guard let expiryString = Keychain.get(KeychainKey.whoopTokenExpiry),
              let expiry = Double(expiryString),
              Date(timeIntervalSince1970: expiry) > Date().addingTimeInterval(60),
              Keychain.get(KeychainKey.whoopAccessToken) != nil else { return false }
        return true
    }

    private func refreshIfNeeded() async throws {
        if hasValidAccessToken() { return }
        if let task = refreshTask {
            try await task.value
            return
        }
        let task = Task<Void, Error> { [self] in try await performRefresh() }
        refreshTask = task
        defer { refreshTask = nil }
        try await task.value
    }

    private func performRefresh() async throws {
        if hasValidAccessToken() { return }
        guard let refresh = Keychain.get(KeychainKey.whoopRefreshToken),
              let secret = Keychain.get(KeychainKey.whoopClientSecret) else {
            throw WhoopError.notConnected
        }
        try await requestToken(form: [
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": Self.clientID,
            "client_secret": secret,
            "scope": "offline",
        ])
    }

    private struct TokenResponse: Decodable {
        let accessToken: String
        let refreshToken: String?
        let expiresIn: Double

        enum CodingKeys: String, CodingKey {
            case accessToken = "access_token"
            case refreshToken = "refresh_token"
            case expiresIn = "expires_in"
        }
    }

    private func requestToken(form: [String: String]) async throws {
        var request = URLRequest(url: Self.tokenURL)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.httpBody = form
            .map { "\($0.key)=\($0.value.addingPercentEncoding(withAllowedCharacters: .alphanumerics) ?? $0.value)" }
            .joined(separator: "&")
            .data(using: .utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            let body = String(data: data, encoding: .utf8) ?? ""
            throw WhoopError.authFailed("token HTTP \(code): \(body.prefix(200))")
        }
        let token = try JSONDecoder().decode(TokenResponse.self, from: data)
        Keychain.set(token.accessToken, for: KeychainKey.whoopAccessToken)
        if let refresh = token.refreshToken {
            Keychain.set(refresh, for: KeychainKey.whoopRefreshToken)
        }
        Keychain.set(String(Date().timeIntervalSince1970 + token.expiresIn),
                     for: KeychainKey.whoopTokenExpiry)
    }

    func disconnect() {
        Keychain.delete(KeychainKey.whoopAccessToken)
        Keychain.delete(KeychainKey.whoopRefreshToken)
        Keychain.delete(KeychainKey.whoopTokenExpiry)
    }

    // MARK: data fetch with backoff + pagination (unchanged HUSTLR mechanics)

    private func get(path: String, query: [URLQueryItem]) async throws -> Data {
        try await refreshIfNeeded()
        guard let token = Keychain.get(KeychainKey.whoopAccessToken) else { throw WhoopError.notConnected }
        var components = URLComponents(url: Self.baseURL.appendingPathComponent(path),
                                       resolvingAgainstBaseURL: false)!
        components.queryItems = query
        var request = URLRequest(url: components.url!)
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

        var attempt = 0
        while true {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else { throw WhoopError.httpError(-1) }
            switch http.statusCode {
            case 200:
                return data
            case 429 where attempt < 4:
                attempt += 1
                try await Task.sleep(for: .seconds(pow(2.0, Double(attempt))))
            case 401 where attempt == 0:
                attempt += 1
                Keychain.delete(KeychainKey.whoopTokenExpiry)
                try await refreshIfNeeded()
                if let newToken = Keychain.get(KeychainKey.whoopAccessToken) {
                    request.setValue("Bearer \(newToken)", forHTTPHeaderField: "Authorization")
                }
            default:
                throw WhoopError.httpError(http.statusCode)
            }
        }
    }

    private func fetchAll<T: Decodable & Sendable>(_ type: T.Type, path: String,
                                                   start: Date, end: Date) async throws -> [T] {
        let iso = ISO8601DateFormatter()
        var all: [T] = []
        var nextToken: String?
        repeat {
            var query = [
                URLQueryItem(name: "limit", value: "25"),
                URLQueryItem(name: "start", value: iso.string(from: start)),
                URLQueryItem(name: "end", value: iso.string(from: end)),
            ]
            if let nextToken { query.append(URLQueryItem(name: "nextToken", value: nextToken)) }
            let data = try await get(path: path, query: query)
            do {
                let page = try decoder.decode(WhoopPage<T>.self, from: data)
                all.append(contentsOf: page.records)
                nextToken = page.nextToken
            } catch {
                throw WhoopError.decodeFailed(path)
            }
        } while nextToken != nil
        return all
    }

    func workouts(start: Date, end: Date) async throws -> [WhoopWorkout] {
        try await fetchAll(WhoopWorkout.self, path: "v2/activity/workout", start: start, end: end)
    }

    /// Pulls the most recent workout of the last 24h and writes it to
    /// Documents/whoop_workout.json for ml/correlate_hr.py --whoop-json.
    func exportLatestWorkout() async throws -> WhoopWorkout? {
        let all = try await workouts(start: Date().addingTimeInterval(-86400), end: Date())
        guard let latest = all.max(by: { ($0.start ?? .distantPast) < ($1.start ?? .distantPast) })
        else { return nil }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("whoop_workout.json")
        try (try encoder.encode(latest)).write(to: url)
        return latest
    }
}

@MainActor
final class WhoopAuthCoordinator: NSObject, ObservableObject, ASWebAuthenticationPresentationContextProviding {
    static let shared = WhoopAuthCoordinator()
    @Published var connected = false
    private var session: ASWebAuthenticationSession?

    func connect() async throws {
        let state = String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(8))
        var components = URLComponents(url: WhoopClient.authURL, resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "client_id", value: WhoopClient.clientID),
            URLQueryItem(name: "redirect_uri", value: WhoopClient.redirectURI),
            URLQueryItem(name: "response_type", value: "code"),
            URLQueryItem(name: "scope", value: WhoopClient.scopes + " offline"),
            URLQueryItem(name: "state", value: state),
        ]

        let callbackURL: URL = try await withCheckedThrowingContinuation { continuation in
            session = ASWebAuthenticationSession(url: components.url!,
                                                callbackURLScheme: "followcam") { [weak self] url, error in
                Task { @MainActor in self?.session = nil }
                if let url {
                    continuation.resume(returning: url)
                } else {
                    continuation.resume(throwing: WhoopError.authFailed(error?.localizedDescription ?? "cancelled"))
                }
            }
            guard let session else {
                continuation.resume(throwing: WhoopError.authFailed("session unavailable"))
                return
            }
            session.presentationContextProvider = self
            session.prefersEphemeralWebBrowserSession = false
            if !session.start() {
                self.session = nil
                continuation.resume(throwing: WhoopError.authFailed("could not start auth session"))
            }
        }

        let items = URLComponents(url: callbackURL, resolvingAgainstBaseURL: false)?.queryItems
        guard items?.first(where: { $0.name == "state" })?.value == state else {
            throw WhoopError.authFailed("state mismatch")
        }
        guard let code = items?.first(where: { $0.name == "code" })?.value else {
            throw WhoopError.authFailed("no code in callback")
        }
        try await WhoopClient.shared.exchangeCode(code)
        connected = true
    }

    nonisolated func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
        ASPresentationAnchor()
    }
}
