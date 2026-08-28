# GrainVDB: Apple Silicon-Native Embedded Vector Store

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-GrainVDB%20Studio-38bdf8.svg)](https://21e8-miner.github.io/grain-vdb/)
[![Platform: macOS](https://img.shields.io/badge/Platform-macOS%20%28Apple%20Silicon%29-brightgreen.svg)]()
[![Metal: v3.0](https://img.shields.io/badge/Metal-v3.0-blue.svg)]()
[![Swift: SPM](https://img.shields.io/badge/Swift-5.9%2B-orange.svg)]()
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()

> 🌐 **Interactive Web Demo:** [https://21e8-miner.github.io/grain-vdb/](https://21e8-miner.github.io/grain-vdb/) — Test sub-millisecond retrieval, topic filtering, and hallucination audits in your browser.

**GrainVDB** is an Apple Silicon-native embedded vector store for local-first AI, computer use agents, and RAG applications. It delivers sub-millisecond local vector retrieval by leveraging Apple's Unified Memory Architecture (UMA), combining hand-tuned ARM NEON CPU vectorization with high-throughput 2D Metal GPU compute shaders.

---

## ⚡ Key Highlights

- **Adaptive Dual-Engine**:
  - **CPU Accelerate / ARM NEON**: **0.108 ms (107 µs)** single-query latency with zero GPU driver dispatch latency.
  - **2D Metal GPU Compute Pipeline**: **~734 queries / sec** peak batch throughput.
- **Unified CUA Agent Memory (Cua Driver Integration)**:
  - Infinite semantic memory + tamper-proof cryptographic audit provenance for Computer Use Agents.
  - **99.76% Token & Cost Reduction**: Replaces 150k token context stuffing with targeted local replay.
  - Non-blocking async ingestion queue for high-FPS visual desktop recordings.
- **True 4KB Page-Aligned Zero-Copy `mmap`**: Memory-map 50GB+ indices in **0.54 ms** directly into unified Metal buffers with zero heap copying.
- **HNSW Approximate Graph Search**: Sub-linear graph traversal for large-scale embedding spaces.
- **Semantic Coherence Topology Audit**: Built-in entropy and cluster connectivity metrics to detect semantic fractures and hallucination risks before prompting LLMs.
- **Multi-Language SDKs**: Native **Python** (`grainvdb`), **Swift** (Swift Package Manager), and **C ABI** (`libgrainvdb.dylib`).
- **Framework Integrations**: First-class drop-in adapters for **LangChain** and **Cua Driver**.

---

## 🤖 Agent Memory: Zero-Latency Replay & Cryptographic Audit

Long-horizon Computer Use Agents (CUAs) running 300+ steps hit the **Context Wall**: stuffing 300 screenshots into an LLM context costs **\$5.00+ per task** and causes catastrophic forgetting.

GrainVDB + Cua Driver provides the solution: **Semantic Visual Memory (GrainVDB) + Non-repudiable Cryptographic Audit (Cua Driver)**.

```
┌────────────────────────────────────────────────────────────────────────┐
│               THE PERFECT COMPUTER USE AGENT (CUA) STACK              │
│                                                                        │
│   ┌───────────────────────────────┐  ┌──────────────────────────────┐  │
│   │   GrainVDB (Semantic Memory)  │  │ Cua Driver (Security Layer)  │  │
│   │  • Sub-millisecond Recall     │  │ • Cryptographic Action Audit │  │
│   │  • Zero-Copy Unified Memory   │  │ • Tamper-proof OS Sandbox    │  │
│   │  • Multimodal State Search    │  │ • Capability Permissioning   │  │
│   └───────────────┬───────────────┘  └──────────────┬───────────────┘  │
│                   │                                 │                  │
│                   └───────────────┬─────────────────┘                  │
│                                   ▼                                    │
│             Zero-Token Replay & Self-Healing Agent Loop                │
│              (99.76% Token Reduction, <$0.01 Cost/Task)                │
└────────────────────────────────────────────────────────────────────────┘
```

### Python Agent Loop Integration
```python
from grainvdb import CuaGrainMemory

# 1. Initialize Unified Memory Layer on Metal GPU
memory = CuaGrainMemory(dim=768)

# 2. Ingest agent action & screenshot embedding asynchronously
memory.record_action_async(
    cua_sequence_id=249,
    semantic_text="macOS Permission Dialog: Filesystem write permission requested.",
    screenshot_embedding=screen_embed,
    app_name="Finder",
    action_type="click"
)

# 3. On failure, perform zero-latency semantic recall
recalled = memory.semantic_recall(query_embedding=error_screen_embed, k=1)
failed_seq = recalled[0]["cua_sequence"]  # Sequence #249 in 0.3ms

# 4. Pull cryptographic audit proof for deterministic LLM correction
audit = memory.secure_audit(failed_seq)
# {"action": "click", "target": "Cancel Button", "outcome": "denied", "proof": "sha256:..."}
```

### Swift Native Mac App Integration
```swift
import GrainVDB

let memory = CuaGrainMemorySwift()
try memory.startMemoryEngine(dimension: 768)

// Record UI State
memory.recordState(cuaSeq: 249, text: "Permission dialog", embedding: screenEmbedding, app: "Finder")

// Semantic Search
if let events = memory.semanticRecall(queryEmbedding: errorEmbedding, k: 1) {
    let failedSeq = events[0].cuaSequence
    let auditProof = try await memory.secureAudit(cuaSequence: failedSeq)
    print("Verified Action: \(auditProof ?? [:])")
}
```

---

## 📊 Performance Benchmarks

### 1. Vector Search (20,000 Vectors @ 128D)
| Execution Backend | Latency (p50) | Throughput / Load Time | Recall |
| :--- | :--- | :--- | :--- |
| **Apple Accelerate / NEON Fast-Path** | **0.108 ms (107.5 µs)** | ~9,250 queries / sec | **100%** (Exact) |
| **Metal GPU Batch Engine** | **1.906 ms** (sync-bound) | **734 queries / sec** | **100%** (Exact) |
| **HNSW Approximate Graph** | **0.735 ms** | Sub-linear | 42% – 95% (ef-tuned) |
| **Page-Aligned Zero-Copy `mmap`** | **0.54 ms** | Instant mapped buffer | N/A |

### 2. 300-Step Computer Use Agent Execution
| Metric | Classic Context Stacking | GrainVDB + Cua Driver Replay | Savings |
| :--- | :--- | :--- | :--- |
| **Context Window Size** | 135,000 tokens | **320 tokens** | **99.76% reduction** |
| **Per-Task Cost** | \$2.03 – \$5.40 | **< \$0.005** | **99.8% cost saved** |
| **Step Replay Latency** | 15–30 seconds | **< 1 millisecond** | **10,000x faster** |
| **Audit Compliance** | ❌ None (Fuzzy logs) | **✅ Cryptographic SHA-256** | Zero-trust verified |

---

## 🛠️ CLI Utilities & Demos

```bash
# 1. Run 60-Second Agent Replay & Audit Interactive Demo
python3 examples/cua_memory_demo.py
# Or with Swift native binary:
swift run CuaMemoryDemo

# 2. Agent Memory CLI Tool
agent-memory search "permission denied dialog box" --k 3
agent-memory audit 249
agent-memory bench --steps 300

# 3. GrainVDB Core Benchmarking
grainvdb bench --n-vectors 20000 --dim 128
grainvdb info knowledge_base.gvdb
grainvdb audit knowledge_base.gvdb --dim 128
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

# Run Unit Tests
python3 -m unittest tests/test_vdb.py
swift test
```

---

## 📜 License
MIT License. Free and open source for local-first AI and agentic software builders.
