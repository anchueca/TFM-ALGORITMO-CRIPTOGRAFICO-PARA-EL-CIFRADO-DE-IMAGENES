#!/usr/bin/env bash
# Script de ejecución rápida para coupled_lyapunov.py (con valores por defecto)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$( cd "$DIR/../.." >/dev/null 2>&1 && pwd )"

if [ -f "$REPO_ROOT/.env/bin/python" ]; then
    PYTHON_BIN="$REPO_ROOT/.env/bin/python"
elif [ -f "$REPO_ROOT/venv/bin/python" ]; then
    PYTHON_BIN="$REPO_ROOT/venv/bin/python"
else
    PYTHON_BIN="python3"
fi

"$PYTHON_BIN" "$DIR/coupled_lyapunov.py" "$@"
