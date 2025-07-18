#!/bin/bash
# run: chmod +x deploy.sh
# ./deploy.sh
# echo "🚀 Starting FastAPI..."
# uvicorn main:app --host 0.0.0.0 --port 8000 &

# echo "⏳ Waiting for server to be ready..."
# sleep 7  # hoặc dùng wait-for-port ở trên

# echo "🌐 Starting Cloudflare Tunnel..."
# cloudflared tunnel --url http://localhost:8000

uvicorn main:app --host 0.0.0.0 --port 8000 &
sleep 2
cloudflared tunnel --url http://localhost:8000
# run: bash run.sh
# #!/bin/bash

# uvicorn main:app --host 0.0.0.0 --port 8000 &
# sleep 2

# cat <<EOL > config.yml
# tunnel: dentalgpt_nv_tlu
# credentials-file: 48e74245-243c-4f6a-b9ca-7250a8cdc26e.json

# ingress:
#   - hostname: api.dentalgpt_nv_tlu.com
#     service: http://localhost:8000
#   - service: http_status:404
# EOL

# cloudflared tunnel --config config.yml run dentalgpt_nv_tlu & wait
# # run: bash run.sh
