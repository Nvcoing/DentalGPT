import threading
from local_server import run_local
from tunnel_ngrok import create_tunnel

if __name__ == "__main__":
    # Tạo tunnel
    auth_token = "2trAEunRvTy9WfZNUVGRt4bMhpy_267Sj5MEej5a1A3pkfrhg"
    public_url = create_tunnel(8000, auth_token)

    # Chạy server
    thread = threading.Thread(target=run_local, daemon=True)
    thread.start()

    # Đợi vô hạn để giữ tiến trình
    thread.join()
