//
// GrainVDB.swift
// GrainVDB - Apple Silicon Native Embedded Vector Store
//

import Foundation
import Accelerate
import CGrainVDB

/// Distance metric for vector search.
public enum GrainDistanceMetric: Int32 {
    case cosine = 0
    case euclidean = 1
    case dot = 2
    case manhattan = 3
}

/// Search algorithm mode.
public enum GrainSearchMode: Int32 {
    case exact = 0
    case hnsw = 1
    case hybrid = 2
}

/// Execution engine backend.
public enum GrainEngineType: Int32 {
    case auto = 0
    case accelerate = 1
    case metal = 2
}

/// Vector search result item.
public struct GrainSearchResult: Identifiable, Sendable {
    public let id: UInt64
    public let score: Float
    public let metadata: [String: Any]?

    public init(id: UInt64, score: Float, metadata: [String: Any]? = nil) {
        self.id = id
        self.score = score
        self.metadata = metadata
    }
}

/// Swift wrapper for GrainVDB native vector store.
public final class GrainVDB {
    private var ctx: OpaquePointer?
    public let dimension: Int
    private var metadataStore: [UInt64: [String: Any]] = [:]
    private let lock = NSRecursiveLock()

    public init(
        dimension: Int = 128,
        mode: GrainSearchMode = .exact,
        distance: GrainDistanceMetric = .cosine,
        engine: GrainEngineType = .auto,
        metallibURL: URL? = nil
    ) throws {
        guard dimension % 4 == 0 else {
            throw NSError(
                domain: "GrainVDB",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Dimension must be a multiple of 4 (got \(dimension))"]
            )
        }

        self.dimension = dimension

        var config = gv2_default_config()
        config.dimension = UInt32(dimension)
        config.mode = gv2_search_mode_t(rawValue: UInt32(mode.rawValue))
        config.distance = gv2_distance_t(rawValue: UInt32(distance.rawValue))
        config.engine = gv2_engine_t(rawValue: UInt32(engine.rawValue))

        let metallibPath = metallibURL?.path ?? Bundle.module.path(forResource: "gv_kernel", ofType: "metallib")
        if let path = metallibPath {
            config.metallib_path = (path as NSString).utf8String
        }

        guard let nativeCtx = gv2_ctx_create(&config) else {
            let errorMsg = gv2_get_error(nil).map { String(cString: $0) } ?? "Unknown error"
            throw NSError(
                domain: "GrainVDB",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create GrainVDB context: \(errorMsg)"]
            )
        }
        self.ctx = nativeCtx
    }

    deinit {
        if let ctx = ctx {
            gv2_ctx_destroy(ctx)
        }
    }

    /// Number of stored vectors.
    public var count: Int {
        lock.lock()
        defer { lock.unlock() }
        guard let ctx = ctx else { return 0 }
        return Int(gv2_vector_count(ctx))
    }

    /// Add vectors with optional IDs and metadata.
    public func addVectors(
        _ vectors: [[Float]],
        ids: [UInt64]? = nil,
        metadata: [[String: Any]]? = nil
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        guard let ctx = ctx else { return }

        let n = vectors.count
        guard n > 0 else { return }

        // Flatten vectors
        var flat: [Float] = []
        flat.reserveCapacity(n * dimension)
        for vec in vectors {
            guard vec.count == dimension else {
                throw NSError(
                    domain: "GrainVDB",
                    code: -3,
                    userInfo: [NSLocalizedDescriptionKey: "Vector dimension mismatch: expected \(dimension), got \(vec.count)"]
                )
            }
            flat.append(contentsOf: vec)
        }

        var assignedIDs: [UInt64]
        if let explicitIDs = ids {
            guard explicitIDs.count == n else {
                throw NSError(
                    domain: "GrainVDB",
                    code: -4,
                    userInfo: [NSLocalizedDescriptionKey: "IDs count mismatch"]
                )
            }
            assignedIDs = explicitIDs
        } else {
            let currentCount = UInt64(gv2_vector_count(ctx))
            assignedIDs = (0..<UInt64(n)).map { currentCount + $0 }
        }

        let success = flat.withUnsafeBufferPointer { flatBuf in
            assignedIDs.withUnsafeBufferPointer { idBuf in
                gv2_add_vectors(ctx, flatBuf.baseAddress, idBuf.baseAddress, UInt32(n))
            }
        }

        guard success else {
            let errorMsg = gv2_get_error(ctx).map { String(cString: $0) } ?? "Failed to add vectors"
            throw NSError(domain: "GrainVDB", code: -5, userInfo: [NSLocalizedDescriptionKey: errorMsg])
        }

        if let metaList = metadata {
            for (idx, id) in assignedIDs.enumerated() {
                if idx < metaList.count {
                    metadataStore[id] = metaList[idx]
                }
            }
        }
    }

    /// Search for top-K nearest neighbors.
    public func search(
        query: [Float],
        k: Int = 10,
        filter: ((UInt64, [String: Any]?) -> Bool)? = nil
    ) throws -> [GrainSearchResult] {
        lock.lock()
        defer { lock.unlock() }
        guard let ctx = ctx else { return [] }
        guard query.count == dimension else {
            throw NSError(
                domain: "GrainVDB",
                code: -6,
                userInfo: [NSLocalizedDescriptionKey: "Query dimension mismatch: expected \(dimension), got \(query.count)"]
            )
        }

        let resultPtr: UnsafeMutablePointer<gv2_search_result_t>?
        if let filterFn = filter {
            // Context wrapper for C callback
            class FilterContext {
                let filter: (UInt64, [String: Any]?) -> Bool
                let metaStore: [UInt64: [String: Any]]
                init(filter: @escaping (UInt64, [String: Any]?) -> Bool, metaStore: [UInt64: [String: Any]]) {
                    self.filter = filter
                    self.metaStore = metaStore
                }
            }
            let filterContext = FilterContext(filter: filterFn, metaStore: self.metadataStore)
            let unmanaged = Unmanaged.passRetained(filterContext)

            let cFilter: gv2_filter_fn = { id, userdata in
                guard let userdata = userdata else { return true }
                let ctx = Unmanaged<FilterContext>.fromOpaque(userdata).takeUnretainedValue()
                return ctx.filter(id, ctx.metaStore[id])
            }

            resultPtr = query.withUnsafeBufferPointer { qBuf in
                gv2_search_filtered(ctx, qBuf.baseAddress, UInt32(k), cFilter, unmanaged.toOpaque())
            }
            unmanaged.release()
        } else {
            resultPtr = query.withUnsafeBufferPointer { qBuf in
                gv2_search(ctx, qBuf.baseAddress, UInt32(k))
            }
        }

        guard let res = resultPtr else {
            let errorMsg = gv2_get_error(ctx).map { String(cString: $0) } ?? "Search failed"
            throw NSError(domain: "GrainVDB", code: -7, userInfo: [NSLocalizedDescriptionKey: errorMsg])
        }
        defer { gv2_free_result(res) }

        var results: [GrainSearchResult] = []
        let numResults = Int(res.pointee.num_results)
        for i in 0..<numResults {
            let id = res.pointee.indices[i]
            let score = res.pointee.scores[i]
            results.append(GrainSearchResult(id: id, score: score, metadata: metadataStore[id]))
        }
        return results
    }

    /// Save database to disk.
    public func save(to path: String) throws {
        lock.lock()
        defer { lock.unlock() }
        guard let ctx = ctx else { return }
        guard gv2_save(ctx, (path as NSString).utf8String) else {
            let errorMsg = gv2_get_error(ctx).map { String(cString: $0) } ?? "Save failed"
            throw NSError(domain: "GrainVDB", code: -8, userInfo: [NSLocalizedDescriptionKey: errorMsg])
        }
    }

    /// Load database from disk.
    public func load(from path: String) throws {
        lock.lock()
        defer { lock.unlock() }
        guard let ctx = ctx else { return }
        guard gv2_load(ctx, (path as NSString).utf8String) else {
            let errorMsg = gv2_get_error(ctx).map { String(cString: $0) } ?? "Load failed"
            throw NSError(domain: "GrainVDB", code: -9, userInfo: [NSLocalizedDescriptionKey: errorMsg])
        }
    }

    /// Open database with zero-copy page-aligned memory mapping.
    public func mmap(from path: String) throws {
        lock.lock()
        defer { lock.unlock() }
        guard let ctx = ctx else { return }
        guard gv2_mmap(ctx, (path as NSString).utf8String) else {
            let errorMsg = gv2_get_error(ctx).map { String(cString: $0) } ?? "mmap failed"
            throw NSError(domain: "GrainVDB", code: -10, userInfo: [NSLocalizedDescriptionKey: errorMsg])
        }
    }
}
