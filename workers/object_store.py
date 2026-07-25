"""Content-addressed object storage for worker outputs (R2 / S3 compatible).

Used by the BiRefNet worker so a cutout is uploaded once and re-served from the
CDN afterwards: the same source image requested twice never runs the model
twice. Falls back to a local directory when no bucket is configured, so the
worker still works on a dev box.

Environment:
    R2_ENDPOINT / S3_ENDPOINT_URL     bucket endpoint
    CLOUDFLARE_R2_ACCESS_KEY_ID       access key
    CLOUDFLARE_R2_SECRET_ACCESS_KEY   secret key
    OMNISERVE_OBJECT_BUCKET           bucket name (default CLOUDFLARE_BUCKET / R2_BUCKET)
    OMNISERVE_OBJECT_PUBLIC_BASE      public URL base (default https://$R2_PUBLIC_HOST)
    OMNISERVE_OBJECT_CACHE_DIR        local fallback directory
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import boto3
    from botocore.client import Config
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover - local-only deployments
    boto3 = None
    Config = None
    ClientError = Exception

ENDPOINT = os.getenv("R2_ENDPOINT") or os.getenv("S3_ENDPOINT_URL", "")
ACCESS_KEY = os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID", "")
SECRET_KEY = os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY", "")
BUCKET = (
    os.getenv("OMNISERVE_OBJECT_BUCKET")
    or os.getenv("CLOUDFLARE_BUCKET")
    or os.getenv("R2_BUCKET", "")
)
PUBLIC_BASE = os.getenv("OMNISERVE_OBJECT_PUBLIC_BASE") or (
    f"https://{os.getenv('R2_PUBLIC_HOST')}" if os.getenv("R2_PUBLIC_HOST") else ""
)
CACHE_DIR = Path(os.getenv("OMNISERVE_OBJECT_CACHE_DIR", "/tmp/omniserve-objects"))

_client = None


def remote_configured() -> bool:
    return bool(boto3 and ENDPOINT and ACCESS_KEY and SECRET_KEY and BUCKET)


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=ENDPOINT,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            config=Config(signature_version="s3v4"),
        )
    return _client


def cache_key(source: str, params: dict[str, Any], *, prefix: str = "cutouts", suffix: str = "webp") -> str:
    """Content key derived from the input URL plus the parameters that change
    the output. Two identical requests therefore collide on purpose."""
    normalised = _normalise_url(source)
    payload = normalised + "|" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"{prefix}/{digest[:2]}/{digest}.{suffix}"


# Signature/expiry parameters only. Matching these by prefix would be wrong:
# "sepia" starts with "se", "svg" with "sv", and folding those away would make
# two genuinely different images share one cache entry.
_SIGNING_PARAMS = {
    "signature", "expires", "token", "sig",       # generic / GCS
    "se", "sp", "sv", "sr", "st", "spr", "sip",   # Azure SAS
    "policy", "key-pair-id",                       # CloudFront
}


def _normalise_url(url: str) -> str:
    """Drops volatile signing parameters so a rotating signed URL for the same
    asset still hits one cache entry, while keeping every parameter that can
    change the bytes we fetch."""
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        return url.strip()
    keep = []
    for part in parsed.query.split("&"):
        if not part:
            continue
        name = part.split("=")[0].lower()
        if name in _SIGNING_PARAMS or name.startswith("x-amz-") or name.startswith("x-goog-"):
            continue
        keep.append(part)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}" + ("?" + "&".join(sorted(keep)) if keep else "")


def public_url(key: str) -> str | None:
    if PUBLIC_BASE:
        return f"{PUBLIC_BASE.rstrip('/')}/{key}"
    return None


def exists(key: str) -> bool:
    """True when the object is already stored - checked before doing any work."""
    if remote_configured():
        try:
            _get_client().head_object(Bucket=BUCKET, Key=key)
            return True
        except ClientError:
            return False
        except Exception:
            return False
    return (CACHE_DIR / key).exists()


def get(key: str) -> bytes | None:
    if remote_configured():
        try:
            response = _get_client().get_object(Bucket=BUCKET, Key=key)
            return response["Body"].read()
        except Exception:
            return None
    path = CACHE_DIR / key
    return path.read_bytes() if path.exists() else None


def put(key: str, data: bytes, content_type: str = "image/webp") -> str | None:
    """Stores the object and returns its public URL (None when only local)."""
    if remote_configured():
        _get_client().put_object(
            Bucket=BUCKET,
            Key=key,
            Body=data,
            ContentType=content_type,
            CacheControl="public, max-age=31536000, immutable",
        )
        return public_url(key)

    path = CACHE_DIR / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return None


def describe() -> dict[str, Any]:
    return {
        "backend": "r2" if remote_configured() else "local",
        "bucket": BUCKET if remote_configured() else str(CACHE_DIR),
        "public_base": PUBLIC_BASE or None,
    }
