#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
#  start_client.sh — Launch the encrypted video streaming client
#
#  Usage:
#    ./start_client.sh [web_port]
#
#  Opens a beautiful web interface in your browser where you can:
#    1. Enter the server IP address and port
#    2. Enter the decryption password
#    3. Click Connect to view the decrypted video stream
#
#  The web UI opens at http://localhost:8080/ by default.
# ══════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${SCRIPT_DIR}/bin/video_client"
FLASK_APP="${SCRIPT_DIR}/client/client_app.py"

# Check binary exists
if [ ! -f "$BIN" ]; then
    echo "❌ Binary not found. Compiling..."
    make -C "$SCRIPT_DIR" client
fi

# Check Python + Flask
if ! python3 -c "import flask" 2>/dev/null; then
    echo "❌ Flask not installed. Install with: pip3 install flask"
    exit 1
fi

WEB_PORT="${1:-8080}"

echo ""
echo "  🔓 Encrypted Video Client"
echo "  ─────────────────────────────────────"
echo "  Web UI:  http://localhost:${WEB_PORT}/"
echo ""
echo "  Opening browser..."
echo ""

# Try to open browser (non-blocking)
(sleep 2 && xdg-open "http://localhost:${WEB_PORT}/" 2>/dev/null || true) &

# Start Flask app
exec python3 "$FLASK_APP"
