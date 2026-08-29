# GrainVDB: Apple Silicon-Native Embedded Vector Store & Agent Trajectory Memory

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/21e8-miner/grain-vdb/actions/workflows/ci.yml/badge.svg)](https://github.com/21e8-miner/grain-vdb/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Concept%20Demo-GrainVDB%20Studio-38bdf8.svg)](https://21e8-miner.github.io/grain-vdb/)
[![Agent DVR](https://img.shields.io/badge/Interactive-Agent%20DVR%20Studio-a855f7.svg)](https://21e8-miner.github.io/grain-vdb/agent_dvr.html)
[![Platform: macOS](https://img.shields.io/badge/Platform-macOS%20%28Apple%20Silicon%29-brightgreen.svg)]()
[![Metal: v3.0](https://img.shields.io/badge/Metal-v3.0-blue.svg)]()
[![Swift: SPM](https://img.shields.io/badge/Swift-5.9%2B-orange.svg)]()
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()

> ⚡ **Reproducible Performance Guarantee:** Every benchmark in this repository is executed and published on every commit by GitHub Actions running on macOS Apple Silicon (`macos-14`) runners. Inspect the latest JSON benchmark runs directly in [CI Artifacts](https://github.com/21e8-miner/grain-vdb/actions/workflows/ci.yml).
>
> 🌐 **Interactive Concept Demo (Simulated in-browser):** [https://21e8-miner.github.io/grain-vdb/](https://21e8-miner.github.io/grain-vdb/)  
> 📹 **Agent DVR Visual Scrubber:** [https://21e8-miner.github.io/grain-vdb/agent_dvr.html](https://21e8-miner.github.io/grain-vdb/agent_dvr.html)

---

## 💡 What is GrainVDB?

**GrainVDB** is an Apple Silicon-native embedded vector store and semantic trajectory memory engine designed for local-first AI, Computer Use Agents (CUAs), and high-throughput on-device RAG.

It leverages Apple's Unified Memory Architecture (UMA) to deliver sub-millisecond local vector retrieval without CPU-to-GPU PCIe transfer penalties, combining:
- **Accelerate / ARM NEON SIMD**: Ultra-low single-query latency (107 µs).
- **2D Metal GPU Compute Pipeline**: High-throughput parallel batch search (734+ QPS).
- **Online Streaming HNSW Graph**: Incremental node insertion ($O(\log N)$) into active layers without rebuilding.
- **Merkle-DAG Trajectory Chaining**: Append-only cryptographic action provenance for tamper-evident agent auditing.
- **Model Context Protocol (MCP) Server**: Drop-in persistent vector memory for Claude Desktop, Cursor, and MCP clients.
- **NumPy Fallback Engine**: Pure Python/NumPy correctness fallback ensuring tests and development run on any OS.

---

## ⚡ Performance Summary

Benchmarks recorded on Apple M-series hardware (20,000 vectors, 128-dim FP16, Top-10 Exact):

| Backend | Single-Query Latency (p50) | Batch Throughput | Recall |
| :--- | :--- | :--- | :--- |
| **Apple Accelerate / NEON Fast-Path** | **0.108 ms (107 µs)** | ~9,250 queries/sec | **100%** (Exact) |
| **Metal GPU 2D Grid Compute** | **1.906 ms** (sync-bound) | **734 queries/sec** | **100%** (Exact) |
| **HNSW Approximate Graph** | **0.735 ms** | Sub-linear traversal | 95%+ (ef-tuned) |
| **Zero-Copy Page-Aligned `mmap`** | **0.54 ms startup** | Direct Metal buffer mapping | N/A |
| **Pure NumPy Reference Fallback** | **~2.5 ms** | CPU single-thread | **100%** (Exact) |

*(Generated on each CI run and uploaded as `benchmark_results.json` artifact).*

---

## 🍎 Native macOS Menu Bar App (`GrainMemory.app`)

For macOS users who want zero-terminal setup and one-click agent memory:

1. **Download:** Unzip `grain-memory-mac-app.zip` (available in [GitHub Releases](https://github.com/21e8-miner/grain-vdb/releases)).
2. **Launch:** Run `GrainMemory.app` — it docks directly into your macOS Menu Bar (`⚡ Grain`).
3. **One-Click Claude Setup:** Click *"Configure Claude Desktop (MCP)"* to automatically register local persistent Metal memory with Claude Desktop.
4. **Visual Agent DVR:** Click *"Open Agent DVR Studio"* to inspect indexed trajectory timelines and Merkle proofs in a native desktop window.

Or compile and package it locally:
```bash
./scripts/build_mac_app.sh
# -> Outputs dist/grain-memory-mac-app.zip containing GrainMemory.app
```

---

## 🔌 Model Context Protocol (MCP) Server

GrainVDB includes a built-in MCP server providing persistent local vector memory for Claude Desktop, Cursor, and Claude Code:

### 1. Launch MCP Server
```bash
grainvdb mcp
# or with standalone entrypoint:
python3 -m grainvdb.mcp_server
```

### 2. Configure Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "grainvdb-memory": {
      "command": "python3",
      "args": ["-m", "grainvdb.mcp_server", "--dim", "768"]
    }
  }
}
```

### Supported MCP Tools:
- `add_memory(text, embedding, app_name, metadata)`: Indexes an agent observation into local Metal memory.
- `semantic_recall(query_embedding, k, app_filter)`: Sub-millisecond vector recall.
- `audit_trajectory(sequence_id)`: Retrieves SHA-256 action metadata and verification proofs.
- `verify_chain_integrity()`: Validates cryptographic continuity across the entire trajectory from Genesis.

---

## 🛠️ Python & Swift SDK Usage

### Python Engine
```python
import numpy as np
from grainvdb import GrainVDB, SearchMode, EngineType

# 1. Initialize Vector Store on Apple Silicon
vdb = GrainVDB(dim=128, mode=SearchMode.EXACT, engine=EngineType.METAL)

# 2. Add vectors with metadata
vectors = np.random.randn(1000, 128).astype(np.float32)
vdb.add_vectors(vectors, metadata=[{"doc_id": i} for i in range(1000)])

# 3. Sub-millisecond Search
query = np.random.randn(128).astype(np.float32)
results = vdb.search(query, k=5)
print(f"Top match ID: {results.indices[0]}, score: {results.scores[0]:.4f}")
```

### Computer Use Agent Trajectory Memory & Merkle Verification
```python
from grainvdb.integrations import CuaGrainMemory

memory = CuaGrainMemory(dim=768)

# Record action & screenshot embedding into Merkle trajectory
memory.record_action(
    cua_sequence_id=42,
    semantic_text="Permission dialog: clicked Allow",
    screenshot_embedding=screen_vec,
    app_name="Finder",
    action_type="click"
)

# Semantic recall
matches = memory.semantic_recall(query_vec, k=1)

# Cryptographic inclusion proof & chain verification
proof = memory.get_merkle_proof(42)
valid, err = memory.verify_trajectory_chain()
assert valid is True
```

### Swift Package Manager Integration
```swift
import GrainVDB

let vdb = try GrainVDB(dimension: 128)
try vdb.add(vectors: vectors)

let results = try vdb.search(query: queryVector, k: 5)
for res in results {
    print("Found vector \(res.id) with score \(res.score)")
}
```

---

## 🚀 Installation & Build

```bash
# Clone & Build Native Metal Core
git clone https://github.com/21e8-miner/grain-vdb.git
cd grain-vdb
./build.sh

# Install Python Package
pip install -e .

# Run Full Test Suite (37+ tests)
python3 -m unittest discover -s tests -p "test_*.py" -v
swift test
```

---

## 📜 License & Inquiries

GrainVDB is dual-licensed under the **MIT License** (Free and open-source) and the **GrainVDB Commercial License** (for proprietary closed-source redistribution and custom Metal compute engineering).

- **Commercial Inquiries:** `licensing@grainvdb.dev`
- **Discussions & Feedback:** [https://github.com/21e8-miner/grain-vdb/discussions](https://github.com/21e8-miner/grain-vdb/discussions)
