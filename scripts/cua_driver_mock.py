#!/usr/bin/env python3
"""
Mock Cua Driver CLI
Simulates the cua-driver CLI command line interfaces for testing and demos.
"""

import sys
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Mock Cua Driver CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Command: history
    history_parser = subparsers.add_parser("history")
    history_subparsers = history_parser.add_subparsers(dest="subcommand")

    # Subcommand: show
    show_parser = history_subparsers.add_parser("show")
    show_parser.add_argument("seq_id", type=int, help="Sequence ID")
    show_parser.add_argument("--json", action="store_true", help="Output JSON format")

    args = parser.parse_args()

    if args.command == "history" and args.subcommand == "show":
        # Formulate mock audit logs
        if args.seq_id == 249:
            data = {
                "cua_sequence": 249,
                "timestamp": "2026-08-28T09:04:12-04:00",
                "action": "click",
                "target": "Cancel Button",
                "capability": "filesystem.write",
                "outcome": "denied",
                "cryptographic_proof": "sha256:8f4c2e6d9b0a1c7d4e5f3a2b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d"
            }
        else:
            data = {
                "cua_sequence": args.seq_id,
                "timestamp": "2026-08-28T09:01:00-04:00",
                "action": "view",
                "target": "Downloads Folder",
                "capability": "filesystem.read",
                "outcome": "allowed",
                "cryptographic_proof": "sha256:1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b"
            }

        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print(f"Cua Sequence: {data['cua_sequence']}")
            print(f"Timestamp:    {data['timestamp']}")
            print(f"Action:       {data['action']} on {data['target']}")
            print(f"Capability:   {data['capability']}")
            print(f"Outcome:      {data['outcome']}")
            print(f"Proof:        {data['cryptographic_proof']}")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
