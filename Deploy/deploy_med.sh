#!/bin/bash
# To run: chmod +x deploy_med.sh && ./deploy_med.sh

set -e  # Dừng nếu có lỗi

echo "🔹 Đang chuẩn bị khởi động các server..."

# 🧹 Dọn cổng nếu bị chiếm
for port in 8000 8080; do
    if lsof -i :$port >/dev/null; then
        echo "⚠️ Dọn cổng $port..."
        kill -9 $(lsof -t -i :$port)
        sleep 1
    fi
done

# 🔹 Chạy LLM Server (port 8000)
echo "🚀 Đang khởi động LLM server (port 8000)..."
cd llmserver
uvicorn local_server:app --host 0.0.0.0 --port 8000 --loop asyncio > ../llm.log 2>&1 &
LLM_PID=$!
cd ..

# 🔹 Chạy backend (port 8080)
echo "🚀 Đang khởi động Backend (port 8080)..."
uvicorn main:app --host 0.0.0.0 --port 8080 > backend.log 2>&1 &
BACKEND_PID=$!

# 🔹 Mở Cloudflare tunnel cho cả hai server
echo "🌐 Mở Cloudflare Tunnel cho LLM (8000)..."
cloudflared tunnel --url http://localhost:8000 > llm_tunnel.log 2>&1 &
CF_LLM_PID=$!

echo "🌐 Mở Cloudflare Tunnel cho Backend (8080)..."
cloudflared tunnel --url http://localhost:8080 > backend_tunnel.log 2>&1 &
CF_BACKEND_PID=$!

# 🧷 Dọn tiến trình khi thoát
trap "echo '🛑 Đang dừng tiến trình...'; kill $LLM_PID $BACKEND_PID $CF_LLM_PID $CF_BACKEND_PID" EXIT

echo "✅ Tất cả server đã khởi động!"
wait
