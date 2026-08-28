import Foundation

public struct AgentMemoryEvent: Codable, Sendable {
    public let cuaSequence: Int
    public let semanticText: String
    public let timestamp: Date

    public init(cuaSequence: Int, semanticText: String, timestamp: Date) {
        self.cuaSequence = cuaSequence
        self.semanticText = semanticText
        self.timestamp = timestamp
    }
}

public class CuaGrainMemorySwift {
    private var db: GrainVDB?
    private var cuaBinaryPath: String = "/usr/local/bin/cua-driver"

    public init() {}

    /// Set custom path to cua-driver binary (useful for mocking / custom installs)
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

    /// Stores agent state alongside the Cua sequence ID
    public func recordState(cuaSeq: Int, text: String, embedding: [Float]) {
        guard let db = db else { return }
        
        let event = AgentMemoryEvent(cuaSequence: cuaSeq, semanticText: text, timestamp: Date())
        
        do {
            let encoder = JSONEncoder()
            // Format dates as ISO8601 for interoperability
            if #available(macOS 10.12, iOS 10.0, *) {
                encoder.dateEncodingStrategy = .iso8601
            }
            let data = try encoder.encode(event)
            if let dict = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
                try db.addVectors([embedding], ids: nil, metadata: [dict])
            }
        } catch {
            print("Failed to write to vector store: \(error)")
        }
    }

    /// Instantly finds past visual states based on a current query
    public func semanticRecall(queryEmbedding: [Float], k: Int = 3) -> [AgentMemoryEvent]? {
        guard let db = db else { return nil }
        
        do {
            let results = try db.search(query: queryEmbedding, k: k)
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
                    print("Failed to decode AgentMemoryEvent: \(error)")
                    return nil
                }
            }
        } catch {
            print("Vector search failed: \(error)")
            return nil
        }
    }

    /// Shells out to Cua Driver to verify the cryptographically secured action log
    public func secureAudit(cuaSequence: Int) async throws -> [String: Any]? {
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
                return try JSONSerialization.jsonObject(with: data) as? [String: Any]
            } else {
                let errData = pipe.fileHandleForReading.readDataToEndOfFile()
                if let errMsg = String(data: errData, encoding: .utf8) {
                    print("Cua process failed with error: \(errMsg)")
                }
            }
        } catch {
            print("Failed to execute Cua audit process: \(error)")
        }
        return nil
    }
}
