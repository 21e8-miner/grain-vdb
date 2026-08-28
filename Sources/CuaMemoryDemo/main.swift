import Foundation
import GrainVDB

@main
struct CuaMemoryDemoApp {
    static func main() async throws {
        print("======================================================================")
        print("  GrainVDB Swift Native Demo: Zero-Latency Agent Replay & Audit")
        print("======================================================================\n")

        let memory = CuaGrainMemorySwift()
        let dim = 128
        try memory.startMemoryEngine(dimension: dim)

        print("[1/4] Ingesting 100 agent interaction steps on Apple Silicon...")
        let t0 = CFAbsoluteTimeGetCurrent()
        for i in 0..<100 {
            var vec = [Float](repeating: 0.0, count: dim)
            vec[i % dim] = 1.0
            let text = (i == 42) ? "macOS Permission Dialog: Filesystem write permission requested." : "Agent Step #\(i): Navigating UI"
            let app = (i % 2 == 0) ? "Finder" : "Terminal"
            memory.recordState(cuaSeq: i, text: text, embedding: vec, app: app)
        }
        let ingestElapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        print("  ✓ Ingested 100 states in \(String(format: "%.2f", ingestElapsed))ms (\(String(format: "%.3f", ingestElapsed/100))ms/step)\n")

        print("[2/4] Executing Semantic Recall for 'Permission Dialog'...")
        var queryVec = [Float](repeating: 0.0, count: dim)
        queryVec[42 % dim] = 0.95
        
        let t1 = CFAbsoluteTimeGetCurrent()
        guard let recalled = memory.semanticRecall(queryEmbedding: queryVec, k: 3) else {
            print("  ✗ Semantic recall failed.")
            return
        }
        let searchElapsed = (CFAbsoluteTimeGetCurrent() - t1) * 1000
        print("  ✓ Semantic recall completed in \(String(format: "%.2f", searchElapsed))ms (Metal GPU)\n")

        for (rank, event) in recalled.enumerated() {
            print("  #\(rank + 1) | Cua Seq ID: \(event.cuaSequence) | App: \(event.app ?? "N/A")")
            print("      Context: \"\(event.semanticText)\"")
        }

        print("\n[3/4] Performing Cryptographic Audit on Sequence #42...")
        // Point to mock script if available
        let mockPath = NSString(string: "./scripts/cua_driver_mock.py").expandingTildeInPath
        if FileManager.default.fileExists(atPath: mockPath) {
            memory.setCuaBinaryPath(mockPath)
        }

        if let auditProof = try await memory.secureAudit(cuaSequence: 42) {
            print("  ✓ Cryptographic Provenance Proof:")
            if let jsonData = try? JSONSerialization.data(withJSONObject: auditProof, options: [.prettyPrinted]),
               let jsonString = String(data: jsonData, encoding: .utf8) {
                print(jsonString)
            }
        }

        print("\n======================================================================")
        print("  SWIFT NATIVE DEMO COMPLETE: ZERO LATENCY + CRYPTOGRAPHIC AUDIT")
        print("======================================================================\n")
    }
}
