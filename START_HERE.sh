#!/bin/bash
# Convenience wrapper for common local commands.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_DIR="$ROOT_DIR/distributed_db"

if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON="$ROOT_DIR/.venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

echo "Distributed Vector Database"
echo ""
echo "Commands:"
echo "  1. Status:    bash START_HERE.sh status"
echo "  2. Examples:  bash START_HERE.sh examples"
echo "  3. API:       bash START_HERE.sh api"
echo "  4. Shell:     bash START_HERE.sh shell"
echo ""

if [ "$1" = "status" ]; then
    cd "$DB_DIR"
    "$PYTHON" run.py status
elif [ "$1" = "examples" ]; then
    cd "$DB_DIR"
    "$PYTHON" run.py examples
elif [ "$1" = "api" ]; then
    echo "Starting API server"
    cd "$DB_DIR"
    "$PYTHON" run.py api
elif [ "$1" = "shell" ]; then
    cd "$DB_DIR"
    "$PYTHON" run.py shell
else
    echo "Usage: bash START_HERE.sh [command]"
    echo ""
    echo "Available commands:"
    echo "  status     - Show system status"
    echo "  examples   - Run example queries"
    echo "  api        - Start REST API server"
    echo "  shell      - Interactive shell"
fi
