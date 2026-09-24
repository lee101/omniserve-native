import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Stub(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps({"path": self.path, "input": data,
                           "authorization": self.headers.get("Authorization")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def main():
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    gateway_port = port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_")}
    env.update(OMNISERVE_NATIVE_SECRET="yue-test", OMNISERVE_NATIVE_MUSIC_UPSTREAM=f"http://127.0.0.1:{upstream.server_port}")
    process = subprocess.Popen([sys.argv[1], "--port", str(gateway_port)], env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{gateway_port}"
    try:
        for _ in range(80):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).close()
                break
            except OSError:
                time.sleep(.1)
        request = urllib.request.Request(base + "/v1/music/generations", data=b'{"style":"folk","lyrics":"Morning"}',
            headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(request, timeout=5)
            raise AssertionError("accepted unauthenticated generation")
        except urllib.error.HTTPError as error:
            assert error.code == 401, error.code
        request.add_header("Authorization", "Bearer yue-test")
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.load(response)
        assert result == {"path": "/v1/music/generations", "input": {"style": "folk", "lyrics": "Morning"},
                          "authorization": "Bearer yue-test"}, result
        print("YuE authenticated native routing passed")
    finally:
        process.terminate()
        process.wait(timeout=10)
        upstream.shutdown()
        upstream.server_close()


if __name__ == "__main__":
    main()
