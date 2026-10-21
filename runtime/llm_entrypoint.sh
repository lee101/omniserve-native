#!/bin/sh
set -eu
DIR=${MODEL_DIR:-}
[ -n "$DIR" ] || { [ -d /runpod-volume ] && DIR=/runpod-volume/models || DIR=/models; }
mkdir -p "$DIR"
F="$DIR/$LLM_FILE"
if [ ! -s "$F" ]; then
    python3 -c 'import http.server,os
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(s): s.send_response(204); s.end_headers()
    def log_message(s,*a): pass
http.server.ThreadingHTTPServer(("0.0.0.0",int(os.environ["PORT"])),H).serve_forever()' &
    PING=$!
    hf download "$LLM_REPO" "$LLM_FILE" --local-dir "$DIR" >/dev/null
    kill "$PING" 2>/dev/null || true
    wait "$PING" 2>/dev/null || true
fi
export OMNISERVE_NATIVE_LLM_GGUF="$F" OMNISERVE_NATIVE_PORT="$PORT"
exec /usr/local/bin/omniserve-native --port "$PORT"
