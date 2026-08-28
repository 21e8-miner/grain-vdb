#!/usr/bin/env python3
"""
GrainVDB Command-Line Interface (CLI)
Provides utilities for benchmarking, inspecting database files, and topology auditing.
"""

import argparse
import sys
import os
import struct
import numpy as np

from .engine import GrainVDB, SearchMode, EngineType, Quantization, DistanceMetric


def cmd_bench(args: argparse.Namespace) -> int:
    from ..benchmark import run_benchmark
    run_benchmark(
        n_vectors=args.n_vectors,
        dim=args.dim,
        n_queries=args.n_queries,
        k=args.k,
    )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    path = args.file
    if not os.path.exists(path):
        print(f"Error: File '{path}' does not exist.", file=sys.stderr)
        return 1

    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        header_bytes = f.read(4096)

    if len(header_bytes) < 24:
        print(f"Error: File '{path}' is too small to be a valid GrainVDB index.", file=sys.stderr)
        return 1

    header = struct.unpack_from("<6I", header_bytes)
    magic = header[0]
    version = header[1]
    dim = header[2]
    quant_val = header[3]
    count = header[4]
    ids_offset = header[5]

    quant_names = {0: "FP32", 1: "FP16 (Half)", 2: "INT8", 3: "BF16"}
    quant_str = quant_names.get(quant_val, f"Unknown ({quant_val})")

    is_page_aligned = (magic == 0x4752414E)
    format_type = "v2.1 (4KB Page-Aligned Zero-Copy)" if is_page_aligned else "v2.0 (Legacy Packed)"

    print("=" * 60)
    print(f"  GrainVDB Index File Info: {os.path.basename(path)}")
    print("=" * 60)
    print(f"  Path:            {os.path.abspath(path)}")
    print(f"  File Size:       {file_size / (1024 * 1024):.2f} MB ({file_size:,} bytes)")
    print(f"  Format Version:  {format_type}")
    print(f"  Vector Count:    {count:,}")
    print(f"  Dimension:       {dim}")
    print(f"  Quantization:    {quant_str}")
    print(f"  Page-Aligned:    {'✓ Yes (0.5ms zero-copy mmap enabled)' if is_page_aligned else '✗ No'}")
    print("=" * 60)
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    path = args.file
    if not os.path.exists(path):
        print(f"Error: File '{path}' not found.", file=sys.stderr)
        return 1

    db = GrainVDB(dim=args.dim, mode=SearchMode.EXACT)
    db.mmap(path)
    count = db.vector_count
    print(f"Auditing semantic topology of {count:,} vectors in {path}...")

    sample_size = min(count, 50)
    indices = list(range(sample_size))
    audit = db.audit(indices)

    print("=" * 60)
    print("  GrainVDB Semantic Topology Audit")
    print("=" * 60)
    print(f"  Sampled Vectors: {sample_size}")
    print(f"  Connectivity:    {audit.connectivity:.4f}")
    print(f"  Coherence:       {audit.coherence:.4f}")
    print(f"  Entropy:         {audit.entropy:.4f}")
    status = "🟢 Semantically Coherent (Low hallucination risk)" if audit.is_semantically_coherent() else "⚠️ Semantic Fracture Detected"
    print(f"  Status:          {status}")
    print("=" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="grainvdb",
        description="GrainVDB: Apple Silicon-Native Vector Store CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # bench
    bench_parser = subparsers.add_parser("bench", help="Run Apple Silicon performance benchmark")
    bench_parser.add_argument("--n-vectors", type=int, default=20000, help="Number of vectors")
    bench_parser.add_argument("--dim", type=int, default=128, help="Vector dimension")
    bench_parser.add_argument("--n-queries", type=int, default=50, help="Number of queries")
    bench_parser.add_argument("--k", type=int, default=10, help="Top-K nearest neighbors")

    # info
    info_parser = subparsers.add_parser("info", help="Inspect a GrainVDB database file")
    info_parser.add_argument("file", type=str, help="Path to .gvdb database file")

    # audit
    audit_parser = subparsers.add_parser("audit", help="Audit semantic topology and coherence of index")
    audit_parser.add_argument("file", type=str, help="Path to .gvdb database file")
    audit_parser.add_argument("--dim", type=int, default=128, help="Vector dimension (default: 128)")

    args = parser.parse_args()

    if args.command == "bench":
        from benchmark import run_benchmark
        run_benchmark(
            n_vectors=args.n_vectors,
            dim=args.dim,
            n_queries=args.n_queries,
            k=args.k,
        )
        return 0
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "audit":
        return cmd_audit(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
