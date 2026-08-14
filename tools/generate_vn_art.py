#!/usr/bin/env python3
"""Generate visual-novel backgrounds and transparent sprites through OmniServe."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image, ImageOps


def request_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None,
                 timeout: int = 360) -> dict[str, Any]:
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read(4096).decode("utf-8", "replace")
        raise RuntimeError(f"{method} {url}: HTTP {error.code}: {detail}") from error


def image_from_value(value: str, timeout: int = 120) -> Image.Image:
    if value.startswith("data:"):
        raw = base64.b64decode(value.split(",", 1)[1], validate=True)
    else:
        with urlopen(value, timeout=timeout) as response:
            raw = response.read(80 << 20)
    return ImageOps.exif_transpose(Image.open(BytesIO(raw))).copy()


def generated_image(payload: dict[str, Any]) -> Image.Image:
    for entry in payload.get("data", []):
        if entry.get("b64_json"):
            return image_from_value("data:image/webp;base64," + entry["b64_json"])
        if entry.get("url"):
            return image_from_value(entry["url"])
    for key in ("image_url", "url", "data_url"):
        if payload.get(key):
            return image_from_value(payload[key])
    raise RuntimeError("generation returned no image")


def post_image(url: str, payload: dict[str, Any], timeout: int = 360) -> Image.Image:
    request = Request(url, data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(80 << 20)
            content_type = response.headers.get_content_type()
    except HTTPError as error:
        detail = error.read(4096).decode("utf-8", "replace")
        raise RuntimeError(f"POST {url}: HTTP {error.code}: {detail}") from error
    if content_type.startswith("image/"):
        return ImageOps.exif_transpose(Image.open(BytesIO(raw))).copy()
    return generated_image(json.loads(raw))


def atomic_save(image: Image.Image, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    image.save(temporary, format="PNG", optimize=True, compress_level=7)
    temporary.replace(target)


def render_background(base: str, prompt: str, seed: int, width: int, height: int) -> Image.Image:
    image = post_image(f"{base}/v1/images/backgrounds", {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": 9,
        "seed": seed,
        "low_priority": True,
        "teleport": True,
    })
    return ImageOps.fit(image.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS)


def render_foreground(base: str, prompt: str, seed: int, width: int, height: int,
                      timeout: int) -> Image.Image:
    queued = request_json(f"{base}/v1/images/foreground-generations/jobs", "POST", {
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "output_format": "webp",
        "decontaminate": True,
        "cache": True,
    }, timeout)
    job_id = queued.get("job_id")
    if not job_id:
        raise RuntimeError(f"foreground generation did not queue: {queued}")
    deadline = time.monotonic() + timeout
    poll = max(float(queued.get("poll_after_ms", 700)) / 1000.0, 0.2)
    while time.monotonic() < deadline:
        time.sleep(poll)
        try:
            job = request_json(f"{base}/v1/images/foreground-generations/jobs/{job_id}", timeout=30)
        except RuntimeError as error:
            if "HTTP 503" in str(error):
                continue
            raise
        if job.get("status") == "done":
            value = job.get("data_url") or job.get("url")
            if not value:
                value = (job.get("cutout") or {}).get("data_url") or (job.get("cutout") or {}).get("url")
            if not value:
                raise RuntimeError(f"foreground job {job_id} returned no cutout")
            return image_from_value(value)
        if job.get("status") == "error":
            raise RuntimeError(f"foreground job {job_id}: {job.get('error', 'failed')}")
    raise RuntimeError(f"foreground job {job_id} timed out after {timeout}s")


def layout_sprite(image: Image.Image, width: int, height: int, centre_x: int,
                  max_width: int, max_height: int) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.point(lambda value: 255 if value >= 8 else 0).getbbox()
    if bbox is None:
        raise RuntimeError("cutout has no foreground alpha")
    subject = rgba.crop(bbox)
    scale = min(max_width / subject.width, max_height / subject.height)
    new_size = (max(1, round(subject.width * scale)), max(1, round(subject.height * scale)))
    subject = subject.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    x = max(0, min(width - subject.width, round(centre_x - subject.width / 2)))
    y = height - subject.height
    canvas.alpha_composite(subject, (x, y))
    return canvas


def validate_sprite(image: Image.Image) -> None:
    if image.mode != "RGBA":
        raise RuntimeError("sprite is not RGBA")
    alpha = image.getchannel("A")
    bbox = alpha.point(lambda value: 255 if value >= 8 else 0).getbbox()
    if bbox is None or bbox[3] < image.height - 2:
        raise RuntimeError("sprite does not stand on the canvas floor")
    coverage = sum(alpha.histogram()[1:]) / (image.width * image.height)
    if not 0.02 <= coverage <= 0.45:
        raise RuntimeError(f"sprite alpha coverage {coverage:.3f} is implausible")
    if any(image.getpixel(point)[3] for point in ((0, 0), (image.width - 1, 0))):
        raise RuntimeError("sprite top corners are not transparent")
    preview = alpha.resize((155, 125), Image.Resampling.BILINEAR)
    pixels = preview.load()
    seen: set[tuple[int, int]] = set()
    components: list[int] = []
    for y in range(preview.height):
        for x in range(preview.width):
            if (x, y) in seen or pixels[x, y] < 24:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            area = 0
            while stack:
                current_x, current_y = stack.pop()
                area += 1
                for point in ((current_x - 1, current_y), (current_x + 1, current_y),
                              (current_x, current_y - 1), (current_x, current_y + 1)):
                    if (0 <= point[0] < preview.width and 0 <= point[1] < preview.height and
                            point not in seen and pixels[point[0], point[1]] >= 24):
                        seen.add(point)
                        stack.append(point)
            components.append(area)
    components.sort(reverse=True)
    if len(components) > 1 and components[1] >= 50:
        outside = sum(components[1:]) / sum(components)
        if outside > 0.12:
            raise RuntimeError(f"sprite contains multiple foreground subjects ({outside:.1%} outside main subject)")


def validate_background(image: Image.Image, width: int, height: int) -> None:
    if image.size != (width, height):
        raise RuntimeError(f"background is {image.width}x{image.height}, expected {width}x{height}")
    if image.mode not in {"RGB", "RGBA"}:
        raise RuntimeError(f"background mode {image.mode} is not RGB/RGBA")


def selected(name: str, only: set[str]) -> bool:
    return not only or name in only or any(name.startswith(prefix) for prefix in only)


def item_target(project: Path, item: dict[str, Any], fallback: str) -> Path:
    relative = Path(item.get("path", fallback))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe art path: {relative}")
    return project / "game" / relative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--gateway-base", default=os.getenv("OMNISERVE_BASE", "http://127.0.0.1:8791"))
    parser.add_argument("--only", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--kinds", choices=("all", "backgrounds", "sprites"), default="all")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    project = args.manifest.parent
    base = args.gateway_base.rstrip("/")
    only = {item.strip() for item in args.only.split(",") if item.strip()}
    background_size = tuple(manifest.get("background_size", [1280, 720]))
    background_source_size = tuple(manifest.get("background_source_size", background_size))
    sprite_source_size = tuple(manifest.get("sprite_source_size", [768, 1024]))
    sprite_canvas = tuple(manifest.get("sprite_canvas", [620, 500]))
    background_style = manifest.get("background_style", "")
    sprite_style = manifest.get("sprite_style", "")
    failures: list[str] = []
    made = 0

    background_items = [item for item in manifest.get("backgrounds", [])
                        if args.kinds != "sprites" and selected(item["name"], only)]
    sprite_items = [item for item in manifest.get("sprites", [])
                    if args.kinds != "backgrounds" and selected(item["name"], only)]
    for item in background_items + sprite_items:
        if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
            parser.error(f"manifest item {item.get('name', '<unknown>')} has an empty prompt")

    def background(item: dict[str, Any]) -> tuple[int, str | None]:
        name = item["name"]
        target = item_target(project, item, f"images/bg/{name}.png")
        if target.exists() and not args.force:
            try:
                with Image.open(target) as existing:
                    validate_background(existing, int(background_size[0]), int(background_size[1]))
                print(f"valid background {name}", flush=True)
            except Exception as error:
                print(f"invalid background {name}: {error}", file=sys.stderr, flush=True)
                return 0, f"background {name}: {error}"
            return 0, None
        print(f"generate background {name}", flush=True)
        if args.dry_run:
            return 0, None
        prompt = ", ".join(part for part in (item["prompt"], background_style) if part)
        for attempt in range(args.retries + 1):
            try:
                image = render_background(base, prompt, int(item.get("seed", 0)) + attempt * 100003,
                                          int(background_source_size[0]), int(background_source_size[1]))
                image = ImageOps.fit(image, (int(background_size[0]), int(background_size[1])),
                                     method=Image.Resampling.LANCZOS)
                validate_background(image, int(background_size[0]), int(background_size[1]))
                atomic_save(image, target)
                return 1, None
            except Exception as error:
                if attempt >= args.retries:
                    print(f"failed background {name}: {error}", file=sys.stderr, flush=True)
                    return 0, f"background {name}: {error}"
                print(f"retry background {name} ({attempt + 1}/{args.retries}): {error}",
                      file=sys.stderr, flush=True)
                time.sleep(min(2 ** attempt * 2, 15))
        return 0, f"background {name}: exhausted retries"

    def sprite(item: dict[str, Any]) -> tuple[int, str | None]:
        name = item["name"]
        target = item_target(project, item, f"images/sprites/{name}.png")
        if target.exists() and not args.force:
            try:
                with Image.open(target) as existing:
                    validate_sprite(existing.convert("RGBA"))
                print(f"valid sprite {name}", flush=True)
            except Exception as error:
                print(f"invalid sprite {name}: {error}", file=sys.stderr, flush=True)
                return 0, f"sprite {name}: {error}"
            return 0, None
        print(f"generate sprite {name}", flush=True)
        if args.dry_run:
            return 0, None
        prompt = ", ".join(part for part in (item["prompt"], sprite_style) if part)
        for attempt in range(args.retries + 1):
            try:
                cutout = render_foreground(base, prompt, int(item.get("seed", 0)) + attempt * 100003,
                                           int(sprite_source_size[0]), int(sprite_source_size[1]),
                                           args.timeout)
                sprite_image = layout_sprite(
                    cutout, int(sprite_canvas[0]), int(sprite_canvas[1]),
                    int(item.get("centre_x", manifest.get("sprite_centre_x", 310))),
                    int(item.get("max_width", manifest.get("sprite_max_width", 230))),
                    int(item.get("max_height", manifest.get("sprite_max_height", 485))))
                validate_sprite(sprite_image)
                atomic_save(sprite_image, target)
                return 1, None
            except Exception as error:
                if attempt >= args.retries:
                    print(f"failed sprite {name}: {error}", file=sys.stderr, flush=True)
                    return 0, f"sprite {name}: {error}"
                print(f"retry sprite {name} ({attempt + 1}/{args.retries}): {error}",
                      file=sys.stderr, flush=True)
                time.sleep(min(2 ** attempt * 2, 15))
        return 0, f"sprite {name}: exhausted retries"

    def run(items: list[dict[str, Any]], operation: Any) -> None:
        nonlocal made
        if not items:
            return
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = [pool.submit(operation, item) for item in items]
            for future in as_completed(futures):
                count, failure = future.result()
                made += count
                if failure:
                    failures.append(failure)

    run(background_items, background)
    run(sprite_items, sprite)

    print(f"generated {made}; failed {len(failures)}", flush=True)
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
