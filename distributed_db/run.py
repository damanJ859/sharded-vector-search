#!/usr/bin/env python3
"""
Main entry point for Distributed Vector Database system.
"""

import sys
import os

# Add distributed_db to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SHARDS_DIR = os.environ.get("DVD_SHARDS_DIR", os.path.join(BASE_DIR, "data", "shards"))
DEFAULT_VECTOR_DIMENSION = int(os.environ.get("DVD_VECTOR_DIMENSION", "1536"))


def run_coordinator_shell():
    """Interactive coordinator shell."""
    from coordinator.coordinator import Coordinator

    print("\n" + "=" * 70)
    print("  Distributed Vector Database - Interactive Shell")
    print("=" * 70 + "\n")
    
    coordinator = Coordinator(DEFAULT_SHARDS_DIR)
    
    import numpy as np
    
    print("Commands:")
    print("  status          - Show system status")
    print("  search <k>      - Search with random vector (k results)")
    print("  help            - Show this help")
    print("  exit            - Exit shell\n")
    
    while True:
        try:
            cmd = input("dvd> ").strip()
            
            if cmd == "exit":
                print("Goodbye!")
                break
            
            elif cmd == "status":
                status = coordinator.get_status()
                print(f"\nSystem Status:")
                print(f"  Shards: {status['num_shards']}")
                print(f"  Total Vectors: {status['total_vectors']}")
                for shard in status['shards']:
                    print(f"    Shard {shard['shard_id']}: {shard['vector_count']} vectors")
                print()
            
            elif cmd.startswith("search"):
                try:
                    k = int(cmd.split()[1]) if len(cmd.split()) > 1 else 5
                    query = np.random.randn(DEFAULT_VECTOR_DIMENSION).astype(np.float32)
                    results = coordinator.search(query, k=k)
                    
                    print(f"\nTop {k} Results:")
                    for i, (shard_id, doc_id, score) in enumerate(results, 1):
                        shard_id_int = int(shard_id)
                        _, doc = coordinator.get_document(shard_id_int, doc_id)
                        print(f"{i}. Score: {score:.4f} | Shard {shard_id} | {doc[:80]}...")
                    print()
                except (ValueError, IndexError):
                    print("Usage: search [k]")
            
            elif cmd == "help":
                print("\nCommands:")
                print("  status          - Show system status")
                print("  search <k>      - Search with random vector (k results)")
                print("  help            - Show this help")
                print("  exit            - Exit shell\n")
            
            else:
                print("Unknown command. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_examples():
    """Run all example queries."""
    print("\nRunning examples...")
    from examples import (
        example_system_overview,
        example_text_search,
        example_random_search,
        example_specific_document_retrieval,
        example_performance_analysis
    )
    
    try:
        example_system_overview()
        example_text_search()
        example_random_search()
        example_specific_document_retrieval()
        example_performance_analysis()
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


def run_api_server():
    """Run Flask API server."""
    print("\nStarting API server...")
    print("Server will be available at http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /health              - Health check")
    print("  GET  /status              - System status")
    print("  POST /search              - Search by text")
    print("  POST /search-vector       - Search by vector")
    print("  GET  /document/<id>/<doc> - Retrieve document\n")
    
    from api import API_HOST, API_PORT, app, init_system

    init_system()
    app.run(debug=False, host=API_HOST, port=API_PORT)


def main():
    parser = argparse.ArgumentParser(
        description="Distributed vector database"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Shell command
    subparsers.add_parser('shell', help='Interactive shell')
    
    # Examples command
    subparsers.add_parser('examples', help='Run example queries')
    
    # API command
    subparsers.add_parser('api', help='Start REST API server')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    if args.command == 'shell':
        run_coordinator_shell()
    elif args.command == 'examples':
        run_examples()
    elif args.command == 'api':
        run_api_server()
    elif args.command == 'status':
        from coordinator.coordinator import Coordinator

        coordinator = Coordinator(DEFAULT_SHARDS_DIR)
        status = coordinator.get_status()
        print("\n" + "=" * 60)
        print("System Status")
        print("=" * 60)
        print(f"Shards: {status['num_shards']}")
        print(f"Total Vectors: {status['total_vectors']}\n")
        for shard in status['shards']:
            print(f"Shard {shard['shard_id']}: {shard['vector_count']} vectors ({shard['vector_count']/status['total_vectors']*100:.1f}%)")
        print()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
