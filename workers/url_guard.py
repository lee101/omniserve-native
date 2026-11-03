"""SSRF guard for fetching URLs that arrive in request bodies.

Only http(s), no userinfo, and every resolved address must be public unless the
host is listed in OMNISERVE_FETCH_ALLOW_HOSTS (comma list) or
OMNISERVE_FETCH_ALLOW_PRIVATE=1. Connections are pinned to the address that was
checked (no DNS-rebinding window), each redirect hop is re-checked, and bodies
are size-capped. Stdlib only so any worker venv can import it.
"""

from __future__ import annotations

import http.client
import ipaddress
import os
import socket
import ssl
from pathlib import Path
from urllib.parse import urljoin, urlsplit

REDIRECTS = {301, 302, 303, 307, 308}
_NAT64 = ipaddress.ip_network("64:ff9b::/96")


class BlockedURL(ValueError):
    pass


class FetchError(ValueError):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


class TooLarge(FetchError):
    pass


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _allowed_hosts() -> set[str]:
    return {h.strip().lower().rstrip(".") for h in os.getenv("OMNISERVE_FETCH_ALLOW_HOSTS", "").split(",") if h.strip()}


def blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        inner = ip.ipv4_mapped or ip.sixtofour or (ip.teredo[1] if ip.teredo else None)
        if inner is None and ip in _NAT64:
            inner = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if inner is not None:
            return blocked_ip(inner)
    return (not ip.is_global or ip.is_multicast or ip.is_reserved or ip.is_unspecified
            or ip.is_loopback or ip.is_link_local or ip.is_private)


def resolve(url: str, *, allow_loopback: bool = False) -> tuple[object, list[str]]:
    parts = urlsplit(str(url).strip())
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.hostname:
        raise BlockedURL("url must be an http(s) URL")
    if parts.username is not None or parts.password is not None:
        raise BlockedURL("credentials in url are not allowed")
    try:
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError as error:
        raise BlockedURL("url has an invalid port") from error
    host = parts.hostname.lower().rstrip(".")
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError) as error:
        raise BlockedURL("url host could not be resolved") from error
    addresses = []
    for info in infos:
        ip = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        if ip not in addresses:
            addresses.append(ip)
    if not addresses:
        raise BlockedURL("url host could not be resolved")
    if not (_truthy("OMNISERVE_FETCH_ALLOW_PRIVATE") or host in _allowed_hosts()):
        for ip in addresses:
            if allow_loopback and ip.is_loopback:
                continue
            if blocked_ip(ip):
                raise BlockedURL("url resolves to a private or reserved address")
    return parts, [str(ip) for ip in addresses]


def check_url(url: str, *, allow_loopback: bool = False) -> str:
    resolve(url, allow_loopback=allow_loopback)
    return str(url).strip()


def _connection(parts, addresses: list[str], timeout: float) -> http.client.HTTPConnection:
    port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    if parts.scheme.lower() == "https":
        conn = http.client.HTTPSConnection(parts.hostname, port, timeout=timeout,
                                           context=ssl.create_default_context())
    else:
        conn = http.client.HTTPConnection(parts.hostname, port, timeout=timeout)

    def pinned(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, *args, **kwargs):
        last: OSError | None = None
        for ip in addresses:
            try:
                return socket.create_connection((ip, address[1]), timeout, source_address)
            except OSError as error:
                last = error
        raise last or OSError("no address to connect to")

    conn._create_connection = pinned
    return conn


def open_url(url: str, *, method: str = "GET", body=None, headers: dict | None = None, timeout: float = 60.0,
             max_redirects: int = 5, allow_loopback: bool = False) -> http.client.HTTPResponse:
    """Returns the final response (caller closes it). Redirects are followed for GET/HEAD only."""
    current = str(url).strip()
    for _ in range(max(0, max_redirects) + 1):
        parts, addresses = resolve(current, allow_loopback=allow_loopback)
        conn = _connection(parts, addresses, timeout)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        try:
            conn.request(method, target, body=body, headers={"User-Agent": "omniserve-fetch/1", **(headers or {})})
            response = conn.getresponse()
        except BaseException:
            conn.close()
            raise
        if response.status in REDIRECTS and method in {"GET", "HEAD"}:
            location = response.getheader("Location")
            response.close()
            conn.close()
            if not location:
                raise FetchError("redirect has no location", response.status)
            current = urljoin(current, location.strip())
            continue
        return response
    raise FetchError("too many redirects")


def _check(response: http.client.HTTPResponse, max_bytes: int) -> None:
    if not 200 <= response.status < 300:
        raise FetchError(f"fetch returned HTTP {response.status}", response.status)
    try:
        declared = int(response.getheader("Content-Length") or 0)
    except ValueError:
        declared = 0
    if declared > max_bytes:
        raise TooLarge(f"response exceeds {max_bytes} bytes", 413)


def fetch_bytes(url: str, max_bytes: int, **kwargs) -> tuple[bytes, str]:
    """(body, content-type) of a 2xx GET, capped at max_bytes."""
    response = open_url(url, **kwargs)
    with response:
        _check(response, max_bytes)
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise TooLarge(f"response exceeds {max_bytes} bytes", 413)
        return data, response.getheader("Content-Type") or ""


def fetch_to_file(url: str, destination: Path, max_bytes: int, **kwargs) -> str:
    """Streams a 2xx GET to destination, capped at max_bytes; returns content-type."""
    response = open_url(url, **kwargs)
    destination = Path(destination)
    with response:
        _check(response, max_bytes)
        size = 0
        try:
            with destination.open("wb") as output:
                while chunk := response.read(1 << 20):
                    size += len(chunk)
                    if size > max_bytes:
                        raise TooLarge(f"response exceeds {max_bytes} bytes", 413)
                    output.write(chunk)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        return response.getheader("Content-Type") or ""
