#!/usr/bin/env bash
# Start the local GPT-SoVITS TTS service that modules/tts.py talks to.
#
# It is a SEPARATE process on purpose: GPT-SoVITS needs torch 2.14/cu126 while
# MuseTalk pins 2.0.1/cu118, so the two cannot share a virtualenv. It also gets
# its own GPU, so synthesis never competes with the renderer for VRAM or SMs.
#
#   scripts/tts_server.sh              # foreground
#   scripts/tts_server.sh > tts.log 2>&1 &   # background
#
# Env overrides: TTS_GPU (default 1), TTS_PORT (default 9880), GPT_SOVITS_DIR.
set -euo pipefail

GPT_SOVITS_DIR="${GPT_SOVITS_DIR:-/data/jiaz/GPT-SoVITS}"
TTS_GPU="${TTS_GPU:-1}"
TTS_PORT="${TTS_PORT:-9880}"

[ -x "$GPT_SOVITS_DIR/.venv/bin/python" ] || {
    echo "No GPT-SoVITS venv at $GPT_SOVITS_DIR/.venv — see README." >&2
    exit 1
}

cd "$GPT_SOVITS_DIR"
# CUDA_VISIBLE_DEVICES, not a device index in the config: the config's "cuda"
# then resolves to this one card and nothing in the process can reach GPU 0.
exec env CUDA_VISIBLE_DEVICES="$TTS_GPU" \
    .venv/bin/python api_v2.py \
        -a 127.0.0.1 -p "$TTS_PORT" \
        -c GPT_SoVITS/configs/tts_infer.yaml
