import Foundation

public struct AgentMemoryEvent: Codable, Sendable {
    public let cuaSequence: Int
    public let semanticText: String
    public let timestamp: Date
    public let app: String?
    public let action: String?
    public let cryptographicHash: String?

    public init(
        cuaSequence: Int, 
        semanticText: String, 
        timestamp: Date = Date(),
        app: String? = nil,
        action: String? = nil,
        cryptographicHash: String? = nil
    ) {
        self.cuaSequence = cuaSequence
        self.semanticText = semanticText
        self.timestamp = timestamp
        self.app = app
        self.action = action
        self.cryptographicHash = cryptographicHash
    }
}

public class CuaGrainMemorySwift {
    private var db: GrainVDB?
    private var cuaBinaryPath: String = "/usr/local/bin/cua-driver"
    private var auditCache: [Int: [String: Any]] = [:]
    private let cacheLock = NSLock()

    public init() {}

    /// Set custom path to cua-driver binary (useful for custom or test environments)
    public func setCuaBinaryPath(_ path: String) {
        self.cuaBinaryPath = path
    }

    /// Initializes the local vector store optimized for Apple Silicon
    public func startMemoryEngine(dimension: Int = 768) throws {
        do {
            db = try GrainVDB(dimension: dimension, mode: .exact)
            print("[Memory] GrainVDB initialized on Apple Silicon.")
        } catch {
            print("[Memory] Failed to initialize GrainVDB: \(error)")
            throw error
        }
    }

    /// Stores agent state alongside the Cua sequence ID and application metadata
    public func recordState(
        cuaSeq: Int, 
        text: String, 
        embedding: [Float],
        app: String? = nil,
        action: String? = nil,
        cryptographicHash: String? = nil
    ) {
        guard let db = db else { return }
        
        let event = AgentMemoryEvent(
            cuaSequence: cuaSeq, 
            semanticText: text, 
            timestamp: Date(),
            app: app,
            action: action,
            cryptographicHash: cryptographicHash
        )
        
        do {
            let encoder = JSONEncoder()
            if #available(macOS 10.12, iOS 10.0, *) {
                encoder.dateEncodingStrategy = .iso8601
            }
            let data = try encoder.encode(event)
            if let dict = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
                try db.addVectors([embedding], ids: nil, metadata: [dict])
            }
        } catch {
            print("[CuaGrainMemorySwift] Failed to write to vector store: \(error)")
        }
    }

    /// Instantly finds past visual states based on a query vector and optional app filter
    public func semanticRecall(
        queryEmbedding: [Float], 
        k: Int = 3,
        appFilter: String? = nil
    ) -> [AgentMemoryEvent]? {
        guard let db = db else { return nil }
        
        do {
            let filterBlock: ((UInt64, [String: Any]?) -> Bool)? = appFilter != nil ? { (_, meta) in
                guard let meta = meta, let app = meta["app"] as? String else { return false }
                return app == appFilter
            } : nil
            
            let results = try db.search(query: queryEmbedding, k: k, filter: filterBlock)
            let decoder = JSONDecoder()
            if #available(macOS 10.12, iOS 10.0, *) {
                decoder.dateDecodingStrategy = .iso8601
            }
            return results.compactMap { result -> AgentMemoryEvent? in
                guard let metaDict = result.metadata else { return nil }
                do {
                    let data = try JSONSerialization.data(withJSONObject: metaDict, options: [])
                    return try decoder.decode(AgentMemoryEvent.self, from: data)
                } catch {
                    print("[CuaGrainMemorySwift] Failed to decode AgentMemoryEvent: \(error)")
                    return nil
                }
            }
        } catch {
            print("[CuaGrainMemorySwift] Vector search failed: \(error)")
            return nil
        }
    }

    private func getCachedAudit(cuaSequence: Int) -> [String: Any]? {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return auditCache[cuaSequence]
    }

    private func setCachedAudit(cuaSequence: Int, json: [String: Any]) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        auditCache[cuaSequence] = json
    }

    /// Shells out to Cua Driver to verify the cryptographically secured action log with in-memory caching
    public func secureAudit(cuaSequence: Int) async throws -> [String: Any]? {
        if let cached = getCachedAudit(cuaSequence: cuaSequence) {
            return cached
        }

        let task = Process()
        task.executableURL = URL(fileURLWithPath: cuaBinaryPath)
        task.arguments = ["history", "show", "\(cuaSequence)", "--json"]
        
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        
        do {
            try task.run()
            task.waitUntilExit()
            
            if task.terminationStatus == 0 {
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    setCachedAudit(cuaSequence: cuaSequence, json: json)
                    return json
                }
            }
        } catch {
            print("[CuaGrainMemorySwift] Failed to execute Cua audit process: \(error)")
        }
        return nil
    }
}
