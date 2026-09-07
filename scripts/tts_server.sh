#!/usr/bin/env bash
# Starts the speech synthesis service that modules/tts.py talks to.
#
# A separate process by necessity: GPT-SoVITS needs torch 2.14/cu126 while
# MuseTalk pins 2.0.1/cu118, and one virtualenv cannot hold both. Giving it its
# own GPU is the additional benefit — synthesis then never competes with the
# renderer for memory or compute.
#
#   scripts/tts_server.sh                      # foreground
#   scripts/tts_server.sh > tts.log 2>&1 &     # background
#
# Overridable: TTS_GPU (default 1), TTS_PORT (default 9880), GPT_SOVITS_DIR.
set -euo pipefail

GPT_SOVITS_DIR="${GPT_SOVITS_DIR:-/data/jiaz/GPT-SoVITS}"
TTS_GPU="${TTS_GPU:-1}"
TTS_PORT="${TTS_PORT:-9880}"

[ -x "$GPT_SOVITS_DIR/.venv/bin/python" ] || {
    echo "No GPT-SoVITS venv at $GPT_SOVITS_DIR/.venv — see README." >&2
    exit 1
}

cd "$GPT_SOVITS_DIR"
# Isolating the GPU through the environment rather than naming a device in the
# config: "cuda" then resolves to this card, and nothing anywhere in the process
# can reach the one the renderer is using.
exec env CUDA_VISIBLE_DEVICES="$TTS_GPU" \
    .venv/bin/python api_v2.py \
        -a 127.0.0.1 -p "$TTS_PORT" \
        -c GPT_SoVITS/configs/tts_infer.yaml
