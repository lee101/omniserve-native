#!/usr/bin/env python3
"""Sibling image models share the embedded lane's admission budget."""
import concurrent.futures
import http.server
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

from test_image_overflow_routing import (CALLER_KEY, free_port, gateway_env,
                                         post, status, wait_busy, wait_ready)

RESPONSE = b'{ "sibling": true, "bytes": "unchanged" }\n'


class Sibling(http.server.BaseHTTPRequestHandler):
    seen = []
    entered = threading.Event()
    release = threading.Event()
    fail = False

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.seen.append((self.path, body, self.headers))
        self.entered.set()
        assert self.release.wait(10)
        if self.fail:
            self.connection.close()
            return
        self.send_response(201)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(RESPONSE)))
        self.send_header('X-Sibling', 'preserved')
        self.end_headers()
        self.wfile.write(RESPONSE)

    def log_message(self, *_args):
        pass


def raw_post(port, path, body, headers=None):
    request = urllib.request.Request(
        f'http://127.0.0.1:{port}{path}', data=body,
        headers={'Content-Type': 'application/json', 'X-API-Key': CALLER_KEY,
                 'X-Omniserve-Tier': 'paid', **(headers or {})})
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.status, response.read(), response.headers


def main():
    binary, stub = os.environ['OMNISERVE_NATIVE_BIN'], os.environ['OMNISERVE_SD_STUB']
    sibling = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Sibling)
    threading.Thread(target=sibling.serve_forever, daemon=True).start()
    port = free_port()
    env = gateway_env(port, stub, sibling.server_port, model='cache-test.gguf')
    for key in list(env):
        if 'OVERFLOW' in key:
            del env[key]
    # Edits must route even when the local model cannot do reference edits.
    env['OMNISERVE_NATIVE_SD_REFERENCE_EDIT'] = '0'
    env['OMNISERVE_NATIVE_IMAGE_MODEL_UPSTREAMS'] = ','.join(
        f'{name}=http://127.0.0.1:{sibling.server_port}/worker'
        for name in ('RA2', 'qwen-image-2.1', 'qwen', 'local'))
    env['OMNISERVE_NATIVE_IMAGE_MODEL_UPSTREAM_SECRET'] = 'sibling-secret'
    env['OMNISERVE_NATIVE_UPSTREAM_TIMEOUT_MS'] = '2000'
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen([binary, '--port', str(port)], env=env, stdout=log, stderr=log)
        try:
            snapshot = wait_ready(port, process)
            assert set(snapshot['image_model_upstreams']) == {'ra2', 'qwen-image-2.1', 'qwen', 'local'}
            body = b'{ "model": "rA2", "prompt": "sibling-only", "extra": [1,2] }'
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                relay = pool.submit(raw_post, port, '/v1/images/generations?secret=caller-secret', body)
                assert Sibling.entered.wait(5)
                snapshot = status(port)
                assert snapshot['admission']['used_slots'] == 1
                local = pool.submit(post, port, '/v1/images/generations',
                                    {'prompt': 'local', 'width': 64, 'height': 64, 'steps': 2},
                                    {'X-API-Key': CALLER_KEY})
                deadline = time.monotonic() + 5
                while status(port)['admission']['waiting']['free'] != 1:
                    assert time.monotonic() < deadline
                    time.sleep(.01)
                assert not local.done()
                Sibling.release.set()
                code, response, headers = relay.result()
                assert (code, response, headers['X-Sibling']) == (201, RESPONSE, 'preserved')
                assert local.result()[0] == 200
            assert status(port)['admission']['used_slots'] == 0
            path, received, headers = Sibling.seen[0]
            assert path == '/worker/v1/images/generations'  # query auth never leaks
            assert received == body
            assert headers['X-API-Key'] == 'sibling-secret'
            assert headers['X-Omniserve-Tier'] == 'paid'
            assert headers.get('Authorization') is None
            assert headers.get('secret') is None
            # Route edits/img2img without local parsing or image decoding.
            for path in ('edits', 'img2img'):
                assert raw_post(port, '/v1/images/' + path, body)[1] == RESPONSE
                assert Sibling.seen[-1][0] == '/worker/v1/images/' + path
            # An external caller cannot forward its claimed paid tier.
            assert raw_post(port, '/v1/images/generations', body,
                            {'X-Forwarded-For': '203.0.113.1'})[0] == 201
            assert Sibling.seen[-1][2]['X-Omniserve-Tier'] == 'free'
            assert Sibling.seen[-1][2].get('X-Forwarded-For') is None
            for model in (None, 'unknown', 'z-image', 'ZIMAGE', 'LOCAL'):
                payload = {'prompt': 'local', 'width': 64, 'height': 64, 'steps': 2}
                if model is not None:
                    payload['model'] = model
                payload['nested'] = {'model': 'ra2'}
                code, response = post(port, '/v1/images/generations', payload, {'X-API-Key': CALLER_KEY})
                assert code == 200 and 'data' in response, (code, response)
            assert len(Sibling.seen) == 4
            assert status(port)['image_model_upstreams']['ra2']['relay_total'] == 4
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics') as response:
                assert b'omniserve_image_model_relay_total{model="ra2"} 4\n' in response.read()
            # A transport failure releases the shared permit too.
            Sibling.fail = True
            try:
                raw_post(port, '/v1/images/generations', body)
                raise AssertionError('expected relay failure')
            except urllib.error.HTTPError as exc:
                assert exc.code == 502
            assert status(port)['admission']['used_slots'] == 0
            print('image model routing: bodies, paths, trust, shared admission and release ok')
        except Exception:
            log.seek(0)
            print(log.read().decode(errors='replace'))
            raise
        finally:
            Sibling.release.set()
            process.terminate()
            process.wait(timeout=10)
            sibling.shutdown()
            sibling.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
