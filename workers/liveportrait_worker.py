#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter

import object_store


UPSTREAM = os.getenv("LIVEPORTRAIT_UPSTREAM", "https://livew.how.nz").rstrip("/")
UPSTREAM_SECRET = os.getenv("LIVEPORTRAIT_UPSTREAM_SECRET", "")
TIMEOUT_SECONDS = float(os.getenv("LIVEPORTRAIT_TIMEOUT_SECONDS", "300"))
CACHE_SIZE = int(os.getenv("LIVEPORTRAIT_CACHE_SIZE", "512"))
CACHE_TTL_SECONDS = int(os.getenv("LIVEPORTRAIT_CACHE_TTL_SECONDS", "86400"))
FORWARD_PROFILE = os.getenv("LIVEPORTRAIT_FORWARD_PROFILE", "0") == "1"
FRAME_KEYS = (
    "eyes_open_mouth_closed",
    "eyes_open_mouth_half",
    "eyes_open_mouth_open",
    "eyes_closed_mouth_closed",
    "eyes_closed_mouth_half",
    "eyes_closed_mouth_open",
)
SHAPE_PROFILES = {
    "natural_v1": {
        "blink": 0.82,
        "mouth_half": 0.32,
        "mouth_open": 0.68,
        "jaw_open": 0.42,
        "lip_rounding": 0.08,
    },
    "subtle_v1": {
        "blink": 0.74,
        "mouth_half": 0.24,
        "mouth_open": 0.54,
        "jaw_open": 0.34,
        "lip_rounding": 0.05,
    },
}


class ExpressionPackRequest(BaseModel):
    image_url: str = Field(min_length=1, max_length=16384)
    shape_profile: str = "natural_v1"


session = requests.Session()
adapter = HTTPAdapter(pool_connections=8, pool_maxsize=32, max_retries=1, pool_block=True)
session.mount("http://", adapter)
session.mount("https://", adapter)
cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
inflight: dict[str, threading.Event] = {}
cache_lock = threading.Lock()


def cache_key(request: ExpressionPackRequest) -> str:
    # Reuse the same signed-URL canonicalisation as the persistent image cache:
    # rotating CDN credentials must not regenerate an identical six-frame pack.
    return object_store.cache_key(
        request.image_url,
        {"schema": "v1", "shape_profile": request.shape_profile},
        prefix="liveportrait",
        suffix="json",
    )


def cached(key: str) -> dict[str, Any] | None:
    now = time.time()
    with cache_lock:
        entry = cache.get(key)
        if not entry:
            return None
        created, result = entry
        if now - created > CACHE_TTL_SECONDS:
            cache.pop(key, None)
            return None
        cache.move_to_end(key)
        return dict(result)


def store(key: str, result: dict[str, Any]) -> None:
    with cache_lock:
        cache[key] = (time.time(), dict(result))
        cache.move_to_end(key)
        while len(cache) > CACHE_SIZE:
            cache.popitem(last=False)


def validate(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict) or not isinstance(result.get("images"), dict):
        raise HTTPException(502, "LivePortrait returned no image map")
    images = result["images"]
    for key in FRAME_KEYS:
        value = images.get(key)
        if not isinstance(value, str) or not (value.startswith("data:image/png;base64,") or value.startswith("https://")):
            raise HTTPException(502, f"LivePortrait returned invalid {key}")
    return result


def run_upstream(request: ExpressionPackRequest) -> dict[str, Any]:
    profile = SHAPE_PROFILES.get(request.shape_profile)
    if profile is None:
        raise HTTPException(400, "unknown shape_profile")
    payload: dict[str, Any] = {"image_url": request.image_url}
    if FORWARD_PROFILE:
        payload["shape_profile"] = profile
    headers = {"content-type": "application/json"}
    if UPSTREAM_SECRET:
        headers["authorization"] = f"Bearer {UPSTREAM_SECRET}"
    try:
        response = session.post(f"{UPSTREAM}/v1/expression-pack", json=payload, headers=headers, timeout=(10, TIMEOUT_SECONDS))
        response.raise_for_status()
        result = validate(response.json())
    except HTTPException:
        raise
    except (requests.RequestException, ValueError) as error:
        raise HTTPException(502, f"LivePortrait upstream failed: {error}") from error
    result["shape_profile"] = request.shape_profile
    return result


def expression_pack(request: ExpressionPackRequest) -> dict[str, Any]:
    key = cache_key(request)
    result = cached(key)
    if result is not None:
        result["cached"] = True
        return result
    with cache_lock:
        event = inflight.get(key)
        leader = event is None
        if leader:
            event = threading.Event()
            inflight[key] = event
    if not leader:
        if not event.wait(TIMEOUT_SECONDS + 15):
            raise HTTPException(504, "LivePortrait coalesced request timed out")
        result = cached(key)
        if result is None:
            raise HTTPException(502, "LivePortrait coalesced request failed")
        result["cached"] = True
        return result
    try:
        result = run_upstream(request)
        store(key, result)
        result["cached"] = False
        return result
    finally:
        with cache_lock:
            inflight.pop(key, None)
            event.set()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    session.close()


app = FastAPI(title="OmniServe LivePortrait worker", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    with cache_lock:
        return {"ready": True, "upstream": UPSTREAM, "cached_packs": len(cache), "inflight": len(inflight), "profiles": list(SHAPE_PROFILES)}


@app.post("/v1/expression-pack")
def create_expression_pack(request: ExpressionPackRequest) -> dict[str, Any]:
    return expression_pack(request)


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("LIVEPORTRAIT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LIVEPORTRAIT_PORT", "9095")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
