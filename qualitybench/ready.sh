#!/usr/bin/env bash
until rg -q "diffusion=qwen|diffusion load failed" "$1"; do sleep 2; done; rg -o "diffusion=.*" "$1" | tail -1
