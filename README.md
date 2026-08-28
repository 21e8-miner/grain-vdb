# GrainVDB: Apple Silicon-Native Embedded Vector Store

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-GrainVDB%20Studio-38bdf8.svg)](https://21e8-miner.github.io/grain-vdb/)
[![Platform: macOS](https://img.shields.io/badge/Platform-macOS%20%28Apple%20Silicon%29-brightgreen.svg)]()
[![Metal: v3.0](https://img.shields.io/badge/Metal-v3.0-blue.svg)]()
[![Swift: SPM](https://img.shields.io/badge/Swift-5.9%2B-orange.svg)]()
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()

> 🌐 **Interactive Web Demo:** [https://21e8-miner.github.io/grain-vdb/](https://21e8-miner.github.io/grain-vdb/) — Test sub-millisecond retrieval, topic filtering, and hallucination audits in your browser.

**GrainVDB** is an Apple Silicon-native embedded vector store for local-first AI and RAG applications. It delivers sub-millisecond local vector retrieval by leveraging Apple's Unified Memory Architecture (UMA), combining hand-tuned ARM NEON CPU vectorization with high-throughput 2D Metal GPU compute shaders.

---

## ⚡ Key Highlights

- **Adaptive Dual-Engine**:
  - **CPU Accelerate / ARM NEON**: **0.108 ms (107 µs)** single-query latency with zero GPU driver dispatch latency.
  - **2D Metal GPU Compute Pipeline**: **~734 queries / sec** peak batch throughput.
- **True 4KB Page-Aligned Zero-Copy `mmap`**: Memory-map 50GB+ indices in **0.54 ms** directly into unified Metal buffers with zero heap copying.
- **HNSW Approximate Graph Search**: Sub-linear graph traversal for large-scale embedding spaces.
- **Semantic Coherence Topology Audit**: Built-in entropy and cluster connectivity metrics to detect semantic fractures and hallucination risks before prompting LLMs.
- **Multi-Language SDKs**: Native **Python** (`grainvdb`), **Swift** (Swift Package Manager), and **C ABI** (`libgrainvdb.dylib`).
- **LangChain Integration**: First-class drop-in `GrainVDBVectorStore` adapter.

---

## 📊 Live Benchmark Metrics (20,000 Vectors @ 128D)

| Execution Backend | Latency (p50) | Throughput / Load Time | Recall |
| :--- | :--- | :--- | :--- |
| **Apple Accelerate / NEON Fast-Path** | **0.108 ms (107.5 µs)** | ~9,250 queries / sec | **100%** (Exact) |
| **Metal GPU Batch Engine** | **1.906 ms** (sync-bound) | **734 queries / sec** | **100%** (Exact) |
| **HNSW Approximate Graph** | **0.735 ms** | Sub-linear | 42% – 95% (ef-tuned) |
| **Page-Aligned Zero-Copy `mmap`** | **0.54 ms** | Instant mapped buffer | N/A |

---

## 🚀 Quickstart (Python)

### 1. Installation & Build
```bash
git clone https://github.com/adamsussman/grain-vdb.git
cd grain-vdb
./build.sh
pip install -e .
```

### 2. Basic In-Process Search
```python
import numpy as np
from grainvdb import GrainVDB, SearchMode, EngineType

# Initialize embedded vector database
db = GrainVDB(dim=128, mode=SearchMode.EXACT, engine=EngineType.AUTO)

# Ingest embeddings
vectors = np.random.randn(10000, 128).astype(np.float32)
metadata = [{"doc_id": i, "category": "tech" if i % 2 == 0 else "finance"} for i in range(10000)]
db.add_vectors(vectors, metadata=metadata)

# Query with metadata predicate filtering
query = np.random.randn(128).astype(np.float32)
results = db.search(
    query, 
    k=5, 
    filter=lambda vid, meta: meta["category"] == "tech"
)

for idx, score in zip(results.indices, results.scores):
    print(f"Match: {idx} | Cosine Similarity: {score:.4f} | Meta: {db.get_metadata(idx)}")
```

### 3. Zero-Copy Persistence
```python
# Save page-aligned binary index
db.save("knowledge_base.gvdb")

# Instant zero-copy open
db_mmap = GrainVDB(dim=128)
db_mmap.mmap("knowledge_base.gvdb")  # Opens in 0.54 ms with zero memory duplication
```

---

## 🦜 LangChain Integration

```python
from grainvdb.integrations import GrainVDBVectorStore
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = GrainVDBVectorStore(embedding=embeddings, dim=768)

vectorstore.add_texts([
    "Apple M2 Ultra features 800 GB/s unified memory bandwidth.",
    "GrainVDB executes zero-copy vector search on Apple Silicon."
])

docs = vectorstore.similarity_search("unified memory bandwidth", k=1)
print(docs[0].page_content)
```

---

## 🍏 Swift Package Integration

Add GrainVDB to your `Package.swift`:
```swift
dependencies: [
    .package(url: "https://github.com/adamsussman/grain-vdb.git", branch: "main")
]
```

Use natively in Swift:
```swift
import GrainVDB

let db = try GrainVDB(dimension: 128, mode: .exact)
try db.addVectors(embeddingMatrix)

let results = try db.search(query: queryEmbedding, k: 5)
for res in results {
    print("Doc ID: \(res.id), Score: \(res.score)")
}
```

---

## 🛠️ CLI Utilities

```bash
# Run Apple Silicon hardware benchmark
grainvdb bench --n-vectors 20000 --dim 128

# Inspect .gvdb index file header & page-alignment
grainvdb info knowledge_base.gvdb

# Run semantic topology & hallucination risk audit
grainvdb audit knowledge_base.gvdb --dim 128
```

---

## 📜 License
MIT License. Free and open source for local-first AI builders.
