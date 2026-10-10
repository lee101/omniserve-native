#!/usr/bin/env python3
"""Rename image files whose extension disagrees with their bytes (the server answers WebP q85 whatever
output_format asks for, so a writer that trusts the request saves WebP under .png).

    python3 fix_ext.py DIR...
"""
import os, sys

def kind(head: bytes) -> str:
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:3] == b"\xff\xd8\xff":
        return "jpg"
    return ""

renamed = 0
for top in sys.argv[1:]:
    for root, _, files in os.walk(top):
        for name in files:
            stem, _, ext = name.rpartition(".")
            if ext.lower() not in ("png", "webp", "jpg", "jpeg"):
                continue
            path = os.path.join(root, name)
            with open(path, "rb") as handle:
                real = kind(handle.read(12))
            if real and real != ext.lower().replace("jpeg", "jpg"):
                os.replace(path, os.path.join(root, f"{stem}.{real}"))
                renamed += 1
print(f"renamed {renamed}")
