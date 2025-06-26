#!/bin/bash
uvicorn main:app --host 0.0.0.0 --port 8000 &
sleep 2
cloudflared tunnel --url http://localhost:8000
# run: bash run.sh