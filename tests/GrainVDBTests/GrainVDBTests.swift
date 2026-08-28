import XCTest
@testable import GrainVDB

final class GrainVDBTests: XCTestCase {
    func testGrainVDBCreationAndSearch() throws {
        let db = try GrainVDB(dimension: 128, mode: .exact)
        XCTAssertEqual(db.count, 0)

        let vector = [Float](repeating: 0.1, count: 128)
        try db.addVectors([vector], ids: [1], metadata: [["doc_id": 1, "text": "hello"]])
        XCTAssertEqual(db.count, 1)

        let results = try db.search(query: vector, k: 1)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, 1)
        XCTAssertEqual(results[0].metadata?["doc_id"] as? Int, 1)
        
        let fetchedMeta = db.getMetadata(for: 1)
        XCTAssertEqual(fetchedMeta?["text"] as? String, "hello")
    }

    func testGrainVDBPersistence() throws {
        let db = try GrainVDB(dimension: 128, mode: .exact)
        let vector = [Float](repeating: 0.2, count: 128)
        try db.addVectors([vector], ids: [42], metadata: [["test_val": "persisted"]])

        let tempPath = NSTemporaryDirectory() + "test_db.gvdb"
        if FileManager.default.fileExists(atPath: tempPath) {
            try? FileManager.default.removeItem(atPath: tempPath)
        }
        if FileManager.default.fileExists(atPath: tempPath + ".meta") {
            try? FileManager.default.removeItem(atPath: tempPath + ".meta")
        }

        try db.save(to: tempPath)
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempPath))
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempPath + ".meta"))

        let dbLoaded = try GrainVDB(dimension: 128, mode: .exact)
        try dbLoaded.load(from: tempPath)
        XCTAssertEqual(dbLoaded.count, 1)

        let results = try dbLoaded.search(query: vector, k: 1)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].metadata?["test_val"] as? String, "persisted")

        // clean up
        try? FileManager.default.removeItem(atPath: tempPath)
        try? FileManager.default.removeItem(atPath: tempPath + ".meta")
    }

    func testCuaGrainMemorySwift() throws {
        let cuaMem = CuaGrainMemorySwift()
        try cuaMem.startMemoryEngine(dimension: 128)
        
        let embedding = [Float](repeating: 0.5, count: 128)
        cuaMem.recordState(cuaSeq: 249, text: "permission denied dialog box", embedding: embedding)
        
        let recalled = cuaMem.semanticRecall(queryEmbedding: embedding, k: 1)
        XCTAssertNotNil(recalled)
        XCTAssertEqual(recalled?.count, 1)
        XCTAssertEqual(recalled?[0].cuaSequence, 249)
        XCTAssertEqual(recalled?[0].semanticText, "permission denied dialog box")
    }
}
