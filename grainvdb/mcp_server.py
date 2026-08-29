"""
grainvdb.mcp_server — Model Context Protocol (MCP) Server for GrainVDB.
Provides local-first, Apple Silicon Metal-accelerated persistent vector memory
and cryptographic trajectory auditing for Claude Desktop, Cursor, and AI agents.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .engine import GrainVDB, SearchMode, EngineType
from .embeddings import FastLocalEmbedding
from .integrations.cua import CuaGrainMemory
from .integrations.cua_merkle import MerkleTrajectoryChain


class GrainVDBMCPServer:
    """
    Stdio-based JSON-RPC 2.0 / MCP Server implementing persistent semantic memory tools.
    """
    PROTOCOL_VERSION = "2024-11-05"

    def __init__(self, dimension: int = 128, engine: str = "auto", db_path: Optional[str] = None):
        self.dimension = dimension
        self.embedder = FastLocalEmbedding(dimension=dimension)
        eng = EngineType.METAL if engine == "metal" else (EngineType.ACCELERATE if engine == "accelerate" else EngineType.AUTO)
        self.memory = CuaGrainMemory(dim=dimension, engine=eng)
        self.db_path = db_path
        if db_path:
            self.memory.load_checkpoint(db_path)

    def handle_initialize(self, request_id: Any) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": self.PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "grainvdb-memory",
                    "version": "2.1.0"
                }
            }
        }

    def handle_tools_list(self, request_id: Any) -> Dict[str, Any]:
        tools = [
            {
                "name": "add_memory",
                "description": "Store an observation, UI action, code snippet, or conversation memory into GrainVDB local Apple Silicon vector memory with cryptographic Merkle provenance.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text content or semantic description of the memory"
                        },
                        "app_name": {
                            "type": "string",
                            "description": "Optional application context (e.g. 'VSCode', 'Terminal', 'Finder')"
                        },
                        "action_type": {
                            "type": "string",
                            "description": "Optional action type (e.g. 'click', 'navigate', 'file_write', 'note')"
                        },
                        "embedding": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Optional precomputed float vector embedding (auto-generated if omitted)"
                        },
                        "sequence_id": {
                            "type": "integer",
                            "description": "Optional explicit sequence ID (auto-incremented if omitted)"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Arbitrary key-value JSON metadata"
                        }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "semantic_recall",
                "description": "Search local Apple Silicon vector memory for semantically relevant past actions, notes, or UI states.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of top results to return (default: 5)"
                        },
                        "app_filter": {
                            "type": "string",
                            "description": "Optional filter by application name"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "audit_trajectory",
                "description": "Retrieve cryptographic SHA-256 Merkle-DAG inclusion proof and action metadata for a specific sequence ID.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sequence_id": {
                            "type": "integer",
                            "description": "The sequence ID to audit"
                        }
                    },
                    "required": ["sequence_id"]
                }
            },
            {
                "name": "verify_chain_integrity",
                "description": "Mathematically verify the cryptographic integrity of the entire trajectory chain from Genesis to latest step.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_memory_stats",
                "description": "Retrieve current memory capacity, vector count, dimension, and active Merkle root hash.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": tools}
        }

    def handle_tool_call(self, request_id: Any, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if name == "add_memory":
                text = args.get("text", "")
                app = args.get("app_name")
                action = args.get("action_type", "note")
                metadata = args.get("metadata", {})
                seq_id = args.get("sequence_id", self.memory.total_records + 1)
                
                if "embedding" in args and args["embedding"]:
                    vec = args["embedding"]
                else:
                    vec = self.embedder.embed_query(text)

                success = self.memory.record_action(
                    cua_sequence_id=seq_id,
                    semantic_text=text,
                    screenshot_embedding=vec,
                    app_name=app,
                    action_type=action,
                    extra_metadata=metadata
                )
                if self.db_path:
                    self.memory.save_checkpoint(self.db_path)

                merkle_proof = self.memory.get_merkle_proof(seq_id)

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({
                                    "status": "success" if success else "failed",
                                    "sequence_id": seq_id,
                                    "node_hash": merkle_proof.get("node_hash"),
                                    "total_records": self.memory.total_records
                                }, indent=2)
                            }
                        ]
                    }
                }

            elif name == "semantic_recall":
                query = args.get("query", "")
                k = int(args.get("k", 5))
                app_filter = args.get("app_filter")
                
                q_vec = self.embedder.embed_query(query)
                results = self.memory.semantic_recall(q_vec, k=k, app_filter=app_filter)

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({
                                    "query": query,
                                    "match_count": len(results),
                                    "results": results
                                }, indent=2)
                            }
                        ]
                    }
                }

            elif name == "audit_trajectory":
                seq_id = int(args.get("sequence_id", 0))
                try:
                    proof = self.memory.get_merkle_proof(seq_id)
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(proof, indent=2)
                                }
                            ]
                        }
                    }
                except KeyError as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32602, "message": str(e)}
                    }

            elif name == "verify_chain_integrity":
                valid, err = self.memory.verify_trajectory_chain()
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({
                                    "chain_valid": valid,
                                    "total_nodes": self.memory.merkle_chain.length,
                                    "root_hash": self.memory.merkle_chain.root_hash,
                                    "error": err
                                }, indent=2)
                            }
                        ]
                    }
                }

            elif name == "get_memory_stats":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({
                                    "total_records": self.memory.total_records,
                                    "dimension": self.dimension,
                                    "root_merkle_hash": self.memory.merkle_chain.root_hash,
                                    "engine": self.memory.db.engine.name if hasattr(self.memory.db.engine, "name") else str(self.memory.db.engine)
                                }, indent=2)
                            }
                        ]
                    }
                }

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Tool '{name}' not found"}
                }

        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": str(e)}
            }

    def process_message(self, message_str: str) -> Optional[Dict[str, Any]]:
        try:
            msg = json.loads(message_str)
        except json.JSONDecodeError:
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}

        method = msg.get("method")
        req_id = msg.get("id")

        if method == "initialize":
            return self.handle_initialize(req_id)
        elif method == "notifications/initialized":
            return None
        elif method == "tools/list":
            return self.handle_tools_list(req_id)
        elif method == "tools/call":
            params = msg.get("params", {})
            name = params.get("name", "")
            args = params.get("arguments", {})
            return self.handle_tool_call(req_id, name, args)
        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        else:
            if req_id is not None:
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method '{method}' not found"}}
            return None

    def run_stdio_loop(self):
        """Runs synchronous stdio JSON-RPC loop."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            resp = self.process_message(line)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="GrainVDB Model Context Protocol (MCP) Server")
    parser.add_argument("--dim", type=int, default=128, help="Vector embedding dimension (default: 128)")
    parser.add_argument("--engine", type=str, default="auto", choices=["auto", "metal", "accelerate"], help="Compute engine")
    parser.add_argument("--db-path", type=str, default=None, help="Optional persistent .gvdb file path")
    args = parser.parse_args()

    server = GrainVDBMCPServer(dimension=args.dim, engine=args.engine, db_path=args.db_path)
    server.run_stdio_loop()


if __name__ == "__main__":
    main()
