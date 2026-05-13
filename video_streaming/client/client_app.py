#!/usr/bin/env python3
"""
client_app.py — Flask wrapper for the video_client C++/CUDA backend.

Serves the web UI and proxies commands/stream to the C++ subprocess.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

# ── Configuration ──────────────────────────────────────────────────────────

BACKEND_PORT = 9090
WEB_PORT = 8080
SCRIPT_DIR = Path(__file__).parent.resolve()
WEB_DIR = SCRIPT_DIR / "web"
BIN_DIR = SCRIPT_DIR.parent / "bin"
BACKEND_BIN = BIN_DIR / "video_client"

app = Flask(__name__, static_folder=str(WEB_DIR))

# ── Backend process management ─────────────────────────────────────────────

backend_process = None
backend_lock = threading.Lock()
backend_status = {"connected": False, "fps": 0.0, "frame_id": 0}


def start_backend():
    """Launch the C++/CUDA backend subprocess."""
    global backend_process
    with backend_lock:
        if backend_process and backend_process.poll() is None:
            return  # Already running

        if not BACKEND_BIN.exists():
            print(f"[ERROR] Backend binary not found: {BACKEND_BIN}")
            print(f"[ERROR] Run 'make client' in the video_streaming directory first.")
            sys.exit(1)

        backend_process = subprocess.Popen(
            [str(BACKEND_BIN), str(BACKEND_PORT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # Let stderr pass through to terminal
            bufsize=1,
            text=True,
        )
        print(f"[Flask] Backend started (PID {backend_process.pid})")

        # Start status reader thread
        threading.Thread(target=_read_status, daemon=True).start()


def _read_status():
    """Read JSON status lines from backend stdout."""
    global backend_status
    try:
        for line in backend_process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                backend_status = json.loads(line)
            except json.JSONDecodeError:
                pass
    except Exception:
        pass


def send_command(cmd: dict):
    """Send a JSON command to the backend via stdin."""
    with backend_lock:
        if backend_process and backend_process.poll() is None:
            try:
                backend_process.stdin.write(json.dumps(cmd) + "\n")
                backend_process.stdin.flush()
            except BrokenPipeError:
                pass


def stop_backend():
    """Gracefully stop the backend."""
    global backend_process
    with backend_lock:
        if backend_process and backend_process.poll() is None:
            try:
                backend_process.stdin.write('{"action":"quit"}\n')
                backend_process.stdin.flush()
            except Exception:
                pass
            backend_process.wait(timeout=5)
            backend_process = None


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(WEB_DIR), "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(WEB_DIR), filename)


@app.route("/api/connect", methods=["POST"])
def api_connect():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    host = data.get("host", "127.0.0.1")
    port = data.get("port", 8554)
    password = data.get("password", "")

    if not password:
        return jsonify({"error": "Password is required"}), 400

    send_command({
        "action": "connect",
        "host": host,
        "port": int(port),
        "password": password,
    })
    return jsonify({"status": "connecting"})


@app.route("/api/disconnect", methods=["POST"])
def api_disconnect():
    send_command({"action": "disconnect"})
    return jsonify({"status": "disconnecting"})


@app.route("/api/status")
def api_status():
    return jsonify(backend_status)


@app.route("/stream")
def stream_proxy():
    """Proxy the MJPEG stream from the C++ backend."""
    import urllib.request

    def generate():
        try:
            with urllib.request.urlopen(f"http://localhost:{BACKEND_PORT}/stream") as resp:
                while True:
                    chunk = resp.read(65536) # Increased chunk size
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            print(f"[Flask] Stream proxy error: {e}")

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    start_backend()

    # Give backend time to start
    time.sleep(1)

    print(f"\n{'═' * 50}")
    print(f"  ENCRYPTED VIDEO CLIENT")
    print(f"  Open in browser: http://localhost:{WEB_PORT}/")
    print(f"{'═' * 50}\n")

    try:
        app.run(host="0.0.0.0", port=WEB_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop_backend()


if __name__ == "__main__":
    main()
