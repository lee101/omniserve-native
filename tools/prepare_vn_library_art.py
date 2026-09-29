#!/usr/bin/env python3
"""Create reproducible OmniServe art plans and deduplicate VN sprite aliases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


IMAGE_RE = re.compile(r'^(\s*image\s+)([^=\n]+?)(\s*=\s*)(["\'])([^"\']+)(["\'].*)$', re.MULTILINE)
CHARACTER_RE = re.compile(
    r'^\s*define\s+([A-Za-z_]\w*)\s*=\s*Character\(\s*(?:_\()?["\']([^"\']+)', re.MULTILINE
)


def seed_for(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def humanize(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").replace("-", " ")).strip()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "art"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def story_context(project: Path, listing: dict[str, Any]) -> str:
    if listing.get("description"):
        description = re.sub(r"\s+", " ", listing["description"]).strip()
    else:
        premise = (project / "premise.md").read_text(errors="replace") if (project / "premise.md").exists() else ""
        description = re.sub(r"\s+", " ", re.sub(r"^#.*$", "", premise, flags=re.MULTILINE)).strip()
    return description[:700]


def is_background(name: str, path: str) -> bool:
    return name.strip().startswith("bg ") or bool(re.search(r"(^|/)bg/", path))


def canonical_sprite_path(tag: str, refs: list[dict[str, Any]], game: Path) -> str:
    existing = next((item["path"] for item in refs if (game / item["path"]).exists()), None)
    if existing:
        return existing
    first = Path(refs[0]["path"])
    parent = first.parent if str(first.parent) != "." else Path("images/sprites")
    return (parent / f"{safe_name(tag)}.png").as_posix()


def character_prompt(tag: str, character_name: str, project: Path, listing: dict[str, Any],
                     context: str) -> str:
    authored = next((item for item in listing.get("characters", [])
                     if isinstance(item, dict) and safe_name(str(item.get("id", ""))) == safe_name(tag)), {})
    details = authored.get("portrait_prompt") or (
        f"adult character {character_name}, visually grounded in this story world: {context[:360]}"
    )
    notes = authored.get("style_notes", "distinctive face, practical story-appropriate clothing, adult proportions")
    numeric_name = (character_name.lower() in
                    {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
                    or bool(re.search(r"\d", character_name)))
    if re.fullmatch(r"student\d+", tag.lower()):
        subject_name = "one anonymous adult student, a single individual"
        details = "one mature academy student in practical story-appropriate clothing"
    else:
        subject_name = (f'the single adult character whose nickname or identifier is "{character_name}" '
                        f'(an identifier, never a requested quantity)' if numeric_name else character_name)
    return (
        f"ONE PERSON ONLY. Create a solo isolated adult character portrait of {subject_name}. "
        f"Use case: illustration-story. Asset type: single game character cutout for {listing.get('title') or humanize(project.name)}. "
        f"Primary request: exactly one adult person, {subject_name}. "
        f"Subject: {details}. Character direction: {notes}. "
        "Composition: three-quarter-length standing portrait, facing the viewer, complete head and torso, "
        "hands visible when practical, generous padding, crisp closed silhouette. "
        "Backdrop: perfectly flat, plain neutral studio background for automatic foreground extraction. "
        "Style: polished painterly visual-novel illustration, mature dramatic fiction, consistent proportions. "
        "Constraints: exactly one person and one body, no companions, no duplicate views, no inset portraits, "
        "no thumbnails, no character sheet, no montage, no cropped head, no extra limbs, no white sticker border, "
        "no coloured outline, no text, no lettering, no logo, no watermark."
    )


def prop_prompt(name: str, project: Path, listing: dict[str, Any]) -> str:
    title = listing.get("title") or humanize(re.sub(r"^\d\d-", "", project.name))
    return (
        f"ONE OBJECT ONLY. A solo {humanize(name)} prop from the mature visual novel {title}. "
        "Complete object, three-quarter product view, practical contemporary materials, polished painterly "
        "visual-novel illustration, crisp closed silhouette, generous padding. Perfectly flat plain neutral "
        "studio background for automatic foreground extraction. No person, no hands, no duplicate objects, "
        "no character sheet, no presentation layout, no white sticker border, no coloured outline, no readable "
        "text, no letters, no numbers, no logo, no watermark."
    )


def background_prompt(name: str, project: Path, listing: dict[str, Any], context: str) -> str:
    title = listing.get("title") or humanize(re.sub(r"^\d\d-", "", project.name))
    period = listing.get("period", "")
    return (
        f"Empty, unoccupied {humanize(name)}, an establishing environment for the mature visual novel {title}. "
        f"Setting and period: {period or 'the time and place naturally implied by the location'}. "
        "Wide 16:9 cinematic environment, eye-level perspective, layered depth, open foreground and centre for "
        "overlaid characters, quieter darker lower quarter for dialogue UI. Polished painterly realism, cohesive "
        "dramatic colour, detailed but readable. Environment only. Absolutely no people, no human silhouettes, "
        "no portraits, no presentation layout, no panels, no text, no letters, no numbers, no signs, no captions, "
        "no logo, no watermark, no interface."
    )


def prepare_project(project: Path, apply: bool) -> dict[str, int]:
    game = project / "game"
    scripts = sorted(game.glob("*.rpy"))
    listing = read_json(project / "listing.json")
    context = story_context(project, listing)
    character_names: dict[str, str] = {}
    for script in scripts:
        for tag, name in CHARACTER_RE.findall(script.read_text(errors="replace")):
            character_names[tag] = name

    all_images: list[dict[str, Any]] = []
    file_text: dict[Path, str] = {}
    for script in scripts:
        text = script.read_text(errors="replace")
        file_text[script] = text
        for match in IMAGE_RE.finditer(text):
            name, path = re.sub(r"\s+", " ", match.group(2)).strip(), match.group(5).strip()
            if not Path(path).suffix:
                continue
            all_images.append({"file": script, "start": match.start(), "end": match.end(),
                               "name": name, "path": path, "match": match})

    backgrounds: dict[str, dict[str, Any]] = {}
    sprites: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in all_images:
        if is_background(item["name"], item["path"]):
            path = item["path"]
            if not (game / path).exists():
                if Path(path).suffix.lower() != ".png":
                    path = str(Path(path).with_suffix(".png"))
                key = path
                backgrounds.setdefault(key, {**item, "path": path})
        else:
            sprites[item["name"].split()[0]].append(item)

    replacements: dict[Path, list[tuple[int, int, str]]] = defaultdict(list)
    sprite_plan: list[dict[str, Any]] = []
    tags = list(sprites)
    count = min(max(len(tags), 1), 4)
    for index, tag in enumerate(tags):
        refs = sprites[tag]
        if all((game / item["path"]).exists() for item in refs):
            continue
        canonical = canonical_sprite_path(tag, refs, game)
        for item in refs:
            if (game / item["path"]).exists():
                continue
            if item["path"] == canonical:
                continue
            match = item["match"]
            replacement = "".join((match.group(1), match.group(2), match.group(3),
                                    match.group(4), canonical, match.group(6)))
            replacements[item["file"]].append((item["start"], item["end"], replacement))
        if not (game / canonical).exists():
            position = index % count
            slot = 0.0 if count == 1 else -0.32 + (0.64 * position / (count - 1))
            sprite_plan.append({
                "name": tag,
                "path": canonical,
                "prompt": (prop_prompt(tag, project, listing) if "/props/" in canonical
                           else character_prompt(tag, character_names.get(tag, humanize(tag)),
                                                 project, listing, context)),
                "seed": seed_for(project.name, "sprite", tag),
                "centre_x": round(310 + slot * 620),
            })

    background_plan = []
    for path, item in sorted(backgrounds.items()):
        source_name = re.sub(r"^bg\s+", "", item["name"])
        background_plan.append({
            "name": safe_name(source_name),
            "path": path,
            "prompt": background_prompt(source_name, project, listing, context),
            "seed": seed_for(project.name, "background", path),
        })
        if item["path"] != item["match"].group(5):
            match = item["match"]
            replacement = "".join((match.group(1), match.group(2), match.group(3),
                                    match.group(4), path, match.group(6)))
            replacements[item["file"]].append((item["start"], item["end"], replacement))

    plan = {
        "version": 1,
        "project": project.name,
        "generator": "OmniServe Z-Image backgrounds plus one-stage Z-Image/BiRefNet foreground jobs",
        "background_size": [1280, 720],
        "background_source_size": [640, 384],
        "sprite_source_size": [512, 768],
        "sprite_canvas": [620, 500],
        "sprite_max_width": 230,
        "sprite_max_height": 485,
        "backgrounds": background_plan,
        "sprites": sprite_plan,
    }
    if apply and (background_plan or sprite_plan or replacements):
        for path, edits in replacements.items():
            text = file_text[path]
            for start, end, replacement in sorted(edits, reverse=True):
                text = text[:start] + replacement + text[end:]
            path.write_text(text)
        (project / "art-plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    return {"backgrounds": len(background_plan), "sprites": len(sprite_plan),
            "rewrites": sum(map(len, replacements.values()))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("--only", default="")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    only = {item.strip() for item in args.only.split(",") if item.strip()}
    totals = {"projects": 0, "backgrounds": 0, "sprites": 0, "rewrites": 0}
    for project in sorted(args.library.glob("[0-9][0-9]-*")):
        if only and project.name not in only:
            continue
        result = prepare_project(project, args.apply)
        if not result["backgrounds"] and not result["sprites"] and not result["rewrites"]:
            continue
        totals["projects"] += 1
        for key in ("backgrounds", "sprites", "rewrites"):
            totals[key] += result[key]
        print(project.name, json.dumps(result, sort_keys=True))
    print(json.dumps(totals, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
