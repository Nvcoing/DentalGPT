import threading
import uvicorn
import nest_asyncio
from pyngrok import ngrok
from app import app

nest_asyncio.apply()

def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Đặt token ngrok của bạn tại đây
ngrok.set_auth_token("2trAEunRvTy9WfZNUVGRt4bMhpy_267Sj5MEej5a1A3pkfrhg")  # Thay bằng token của bạn
public_url = ngrok.connect(8000).public_url
print(f"Public URL: {public_url}")

threading.Thread(target=run, daemon=True).start()

# Giữ cho chương trình chạy liên tục
import time
while True:
    time.sleep(1)
