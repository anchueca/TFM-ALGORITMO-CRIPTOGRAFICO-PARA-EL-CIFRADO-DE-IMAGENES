#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
#  start_server.sh — Launch the encrypted video streaming server
#
#  Usage:
#    ./start_server.sh <password> [port] [mjpeg_port] [device] [resolution]
#
#  Examples:
#    ./start_server.sh mySecretKey
#    ./start_server.sh mySecretKey 8554 8555 0 640x480
#
#  The server streams encrypted video on two ports:
#    - TCP port (default 8554): for the custom decryption client
#    - MJPEG port (default 8555): for VLC (shows encrypted noise)
#
#  To view encrypted stream in VLC:
#    vlc http://<server_ip>:8555/
# ══════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${SCRIPT_DIR}/bin/video_server"

# Check binary exists
if [ ! -f "$BIN" ]; then
    echo "❌ Binary not found. Compiling..."
    make -C "$SCRIPT_DIR" server
fi

# Check arguments
if [ -z "$1" ]; then
    echo ""
    echo "  🔐 Encrypted Video Streaming Server"
    echo "  ─────────────────────────────────────"
    echo ""
    echo "  Usage: $0 <password> [port] [mjpeg_port] [device] [resolution]"
    echo ""
    echo "  Arguments:"
    echo "    password     Encryption key (required)"
    echo "    port         TCP port for client connections (default: 8554)"
    echo "    mjpeg_port   HTTP MJPEG port for VLC viewing (default: 8555)"
    echo "    device       Webcam device index (default: 0)"
    echo "    resolution   Capture resolution WxH (default: 640x480)"
    echo ""
    echo "  Example:"
    echo "    $0 mySecretKey123"
    echo "    $0 mySecretKey123 8554 8555 0 1280x720"
    echo ""
    exit 1
fi

PASSWORD="$1"
PORT="${2:-8554}"
MJPEG_PORT="${3:-8555}"
DEVICE="${4:-0}"
RESOLUTION="${5:-640x480}"

echo ""
echo "  🔐 Starting Encrypted Video Server"
echo "  ─────────────────────────────────────"
echo "  Password:    ********"
echo "  TCP Port:    ${PORT} (for client)"
echo "  MJPEG Port:  ${MJPEG_PORT} (for VLC: vlc http://localhost:${MJPEG_PORT}/)"
echo "  Device:      /dev/video${DEVICE}"
echo "  Resolution:  ${RESOLUTION}"
echo ""

exec "$BIN" \
    --password "$PASSWORD" \
    --port "$PORT" \
    --mjpeg-port "$MJPEG_PORT" \
    --device "$DEVICE" \
    --resolution "$RESOLUTION"
