#!/bin/bash

echo "[Avatar API] Waiting for WebUI API..."

while ! nc -z localhost 7860 ; do echo "[Avatar API] Waiting..." && sleep 1 ; done

echo "[Avatar API] WebUI API Available! Running Avatar API..."

export POSTHOG_KEY="$1"
python3 -m fastapi run /avatar_api.py