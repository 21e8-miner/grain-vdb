"""
test_mcp.py — Unit tests for GrainVDB Model Context Protocol (MCP) Server.
"""

import json
import os
import sys
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from grainvdb.mcp_server import GrainVDBMCPServer


class TestGrainVDBMCPServer(unittest.TestCase):

    def setUp(self):
        self.server = GrainVDBMCPServer(dimension=16, engine="accelerate")

    def tearDown(self):
        self.server.memory.close()

    def test_initialize(self):
        msg = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        resp = self.server.process_message(msg)
        self.assertEqual(resp["id"], 1)
        self.assertIn("protocolVersion", resp["result"])
        self.assertEqual(resp["result"]["serverInfo"]["name"], "grainvdb-memory")

    def test_tools_list(self):
        msg = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        resp = self.server.process_message(msg)
        tools = resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        self.assertIn("add_memory", tool_names)
        self.assertIn("semantic_recall", tool_names)
        self.assertIn("audit_trajectory", tool_names)
        self.assertIn("verify_chain_integrity", tool_names)
        self.assertIn("get_memory_stats", tool_names)

    def test_add_and_recall_memory_flow(self):
        # 1. Add memory
        add_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "add_memory",
                "arguments": {
                    "text": "User prefers dark mode and Python 3.12",
                    "app_name": "VSCode",
                    "action_type": "user_preference",
                    "sequence_id": 101
                }
            }
        })
        add_resp = self.server.process_message(add_msg)
        add_content = json.loads(add_resp["result"]["content"][0]["text"])
        self.assertEqual(add_content["status"], "success")
        self.assertEqual(add_content["sequence_id"], 101)

        # 2. Semantic recall
        recall_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "semantic_recall",
                "arguments": {
                    "query": "What are user theme preferences?",
                    "k": 1
                }
            }
        })
        recall_resp = self.server.process_message(recall_msg)
        recall_content = json.loads(recall_resp["result"]["content"][0]["text"])
        self.assertEqual(recall_content["match_count"], 1)
        self.assertEqual(recall_content["results"][0]["cua_sequence"], 101)

        # 3. Audit trajectory
        audit_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "audit_trajectory",
                "arguments": {"sequence_id": 101}
            }
        })
        audit_resp = self.server.process_message(audit_msg)
        audit_content = json.loads(audit_resp["result"]["content"][0]["text"])
        self.assertEqual(audit_content["sequence_id"], 101)
        self.assertEqual(audit_content["tamper_evident_status"], "VERIFIED_VALID")

        # 4. Verify chain integrity
        verify_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "verify_chain_integrity",
                "arguments": {}
            }
        })
        verify_resp = self.server.process_message(verify_msg)
        verify_content = json.loads(verify_resp["result"]["content"][0]["text"])
        self.assertTrue(verify_content["chain_valid"])
        self.assertEqual(verify_content["total_nodes"], 1)


if __name__ == "__main__":
    unittest.main()
