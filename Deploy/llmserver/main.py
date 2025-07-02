import threading
import uvicorn
from pyngrok import ngrok
import nest_asyncio

from app import app

# Cho phép nested async loop khi dùng trong Jupyter hoặc thread
nest_asyncio.apply()

# 1. Thiết lập ngrok và tạo địa chỉ công khai
ngrok.set_auth_token("2trAEunRvTy9WfZNUVGRt4bMhpy_267Sj5MEej5a1A3pkfrhg")  # Thay bằng token thật
public_url = ngrok.connect(8000, bind_tls=True).public_url

print(f"\n🚀 Local API:  http://localhost:8000")
print(f"🌍 Public API: {public_url}\n")

# 2. Khởi chạy Uvicorn ở luồng khác để không bị block
def run_local_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

threading.Thread(target=run_local_server, daemon=True).start()

# 3. Giữ script chạy
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Đã dừng server.")
