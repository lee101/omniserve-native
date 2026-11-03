#!/usr/bin/env python3
"""SSRF guard for request-supplied URLs. No network: DNS and connects are faked,
except one loopback server used to prove redirects are re-checked per hop."""

import os
import socket
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

import url_guard  # noqa: E402

REAL_GETADDRINFO = socket.getaddrinfo


def fake_dns(mapping):
    def resolve(host, port, *args, **kwargs):
        if host not in mapping:
            raise socket.gaierror("unknown host")
        ips = mapping[host] if isinstance(mapping[host], list) else [mapping[host]]
        out = []
        for ip in ips:
            family = socket.AF_INET6 if ":" in ip else socket.AF_INET
            sockaddr = (ip, port, 0, 0) if family == socket.AF_INET6 else (ip, port)
            out.append((family, socket.SOCK_STREAM, 6, "", sockaddr))
        return out
    return mock.patch.object(url_guard.socket, "getaddrinfo", side_effect=resolve)


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("OMNISERVE_FETCH_")}
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


class CheckUrlTest(unittest.TestCase):
    BLOCKED = ["127.0.0.1", "10.1.2.3", "172.16.5.5", "192.168.0.9", "169.254.169.254", "100.64.0.1",
               "0.0.0.0", "224.0.0.1", "::1", "fd00::1", "fe80::1", "::ffff:127.0.0.1", "64:ff9b::a9fe:a9fe",
               "2002:a9fe:a9fe::1", "::"]

    def test_private_and_reserved_addresses_are_refused(self):
        for ip in self.BLOCKED:
            with self.subTest(ip=ip), clean_env(), fake_dns({"evil.example": ip}):
                with self.assertRaises(url_guard.BlockedURL):
                    url_guard.check_url("https://evil.example/a.png")

    def test_any_private_answer_blocks(self):
        with clean_env(), fake_dns({"mixed.example": ["93.184.216.34", "10.0.0.1"]}):
            with self.assertRaises(url_guard.BlockedURL):
                url_guard.check_url("http://mixed.example/")

    def test_literal_ips_and_bad_schemes(self):
        with clean_env():
            for url in ("http://127.0.0.1/", "http://[::1]:8080/", "http://169.254.169.254/latest/meta-data",
                        "file:///etc/passwd", "gopher://x.example/", "ftp://x.example/", "http:///nohost",
                        "http://user:pw@cdn.example/a.png", "http://cdn.example:99999/"):
                with self.subTest(url=url), fake_dns({"cdn.example": "93.184.216.34"}):
                    with self.assertRaises(url_guard.BlockedURL):
                        url_guard.check_url(url)

    def test_public_host_passes(self):
        with clean_env(), fake_dns({"cdn.example": ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"]}):
            self.assertEqual(url_guard.check_url(" https://cdn.example/a.png "), "https://cdn.example/a.png")

    def test_unresolvable_host_is_refused(self):
        with clean_env(), fake_dns({}):
            with self.assertRaises(url_guard.BlockedURL):
                url_guard.check_url("https://nowhere.example/")

    def test_allow_hosts_and_allow_private_env(self):
        with fake_dns({"minio.internal": "10.0.0.5", "other.internal": "10.0.0.6"}):
            with clean_env(OMNISERVE_FETCH_ALLOW_HOSTS="minio.internal, x.example"):
                url_guard.check_url("http://MINIO.internal./bucket/a.png")
                with self.assertRaises(url_guard.BlockedURL):
                    url_guard.check_url("http://other.internal/")
            with clean_env(OMNISERVE_FETCH_ALLOW_PRIVATE="1"):
                url_guard.check_url("http://other.internal/")

    def test_allow_loopback_admits_only_loopback(self):
        with clean_env(), fake_dns({"lo.example": "127.0.0.1", "lan.example": "10.0.0.1"}):
            url_guard.check_url("http://lo.example/", allow_loopback=True)
            with self.assertRaises(url_guard.BlockedURL):
                url_guard.check_url("http://lan.example/", allow_loopback=True)


class PinnedConnectTest(unittest.TestCase):
    def test_connects_to_the_checked_address_not_a_fresh_lookup(self):
        seen = []

        def connect(address, *args, **kwargs):
            seen.append(address)
            raise ConnectionRefusedError("stop here")

        with clean_env(), fake_dns({"cdn.example": "93.184.216.34"}), \
                mock.patch.object(url_guard.socket, "create_connection", side_effect=connect):
            with self.assertRaises(ConnectionRefusedError):
                url_guard.fetch_bytes("http://cdn.example:8080/a.png", 1024)
        self.assertEqual(seen, [("93.184.216.34", 8080)])


class Handler(BaseHTTPRequestHandler):
    routes = {}

    def do_GET(self):
        status, headers, body = self.routes.get(self.path, (404, {}, b""))
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class LoopbackFetchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        Handler.routes = {
            "/ok": (200, {"Content-Type": "image/png"}, b"png-bytes"),
            "/big": (200, {}, b"x" * 2048),
            "/to-metadata": (302, {"Location": "http://169.254.169.254/latest/meta-data/"}, b""),
            "/to-ok": (302, {"Location": "/ok"}, b""),
            "/loop": (302, {"Location": "/loop"}, b""),
            "/missing": (404, {}, b"nope"),
        }

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_loopback_blocked_by_default(self):
        with clean_env():
            with self.assertRaises(url_guard.BlockedURL):
                url_guard.fetch_bytes(self.base + "/ok", 1024)

    def test_fetch_follows_safe_redirect(self):
        with clean_env():
            data, content_type = url_guard.fetch_bytes(self.base + "/to-ok", 1024, allow_loopback=True)
        self.assertEqual((data, content_type), (b"png-bytes", "image/png"))

    def test_redirect_to_metadata_is_rechecked(self):
        with clean_env():
            with self.assertRaises(url_guard.BlockedURL):
                url_guard.fetch_bytes(self.base + "/to-metadata", 1024, allow_loopback=True)

    def test_redirect_loop_and_size_and_status(self):
        with clean_env():
            with self.assertRaises(url_guard.FetchError):
                url_guard.fetch_bytes(self.base + "/loop", 1024, allow_loopback=True)
            with self.assertRaises(url_guard.TooLarge):
                url_guard.fetch_bytes(self.base + "/big", 1024, allow_loopback=True)
            with self.assertRaises(url_guard.FetchError) as caught:
                url_guard.fetch_bytes(self.base + "/missing", 1024, allow_loopback=True)
            self.assertEqual(caught.exception.status, 404)

    def test_fetch_to_file_cleans_up_on_overflow(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder, clean_env():
            target = Path(folder) / "out"
            with self.assertRaises(url_guard.TooLarge):
                url_guard.fetch_to_file(self.base + "/big", target, 1024, allow_loopback=True)
            self.assertFalse(target.exists())
            self.assertEqual(url_guard.fetch_to_file(self.base + "/ok", target, 1024, allow_loopback=True), "image/png")
            self.assertEqual(target.read_bytes(), b"png-bytes")

    def test_allow_hosts_env_admits_loopback_host(self):
        with clean_env(OMNISERVE_FETCH_ALLOW_HOSTS="127.0.0.1"):
            self.assertEqual(url_guard.fetch_bytes(self.base + "/ok", 1024)[0], b"png-bytes")


if __name__ == "__main__":
    unittest.main(verbosity=2)
