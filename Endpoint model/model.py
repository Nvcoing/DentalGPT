import nest_asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from unsloth import FastLanguageModel
import torch
from pyngrok import ngrok
import threading
import asyncio
from transformers import TextStreamer
import gc

# Giúp chạy asyncio trong notebook (Kaggle/Jupyter)
nest_asyncio.apply()

app = FastAPI()

# Load model + tokenizer
model_name = "NV9523/DentalGPT"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True
)
FastLanguageModel.for_inference(model)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Đưa model lên device ngay sau load
model.to(device)
model.eval()  # chế độ inference

@app.post("/model/generate/")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    max_new_tokens = data.get("max_new_tokens", 50)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Tạo một queue để trao đổi dữ liệu giữa TextStreamer và StreamingResponse
    response_queue = asyncio.Queue()
    
    class CustomStreamer(TextStreamer):
        def __init__(self, tokenizer, queue, **kwargs):
            super().__init__(tokenizer, **kwargs)
            self.queue = queue
        
        def on_finalized_text(self, text: str, stream_end: bool = False):
            # Đẩy text vào queue khi có token mới
            asyncio.run_coroutine_threadsafe(self.queue.put(text), loop)
            if stream_end:
                asyncio.run_coroutine_threadsafe(self.queue.put(None), loop)
    
    # Lấy event loop hiện tại
    loop = asyncio.get_event_loop()
    
    # Khởi tạo streamer
    streamer = CustomStreamer(tokenizer, response_queue, skip_prompt=True)
    
    # Hàm generator để yield text từ queue
    async def response_generator():
        while True:
            text = await response_queue.get()
            if text is None:
                break
            yield text
            await asyncio.sleep(0.01)  # Giảm tải CPU
    
    # Chạy generation trong thread riêng để không block event loop
    def generate_in_thread():
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.empty_cache()
        gc.collect()
    
    threading.Thread(target=generate_in_thread, daemon=True).start()
    
    return StreamingResponse(response_generator(), media_type="text/markdown")

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

ngrok.set_auth_token("2trAEunRvTy9WfZNUVGRt4bMhpy_267Sj5MEej5a1A3pkfrhg")
# Mở ngrok tunnel tới port 8000
public_url = ngrok.connect(8000).public_url
print(f"Public URL: {public_url}")

# Chạy FastAPI server trong thread riêng để không block cell
thread = threading.Thread(target=run_api, daemon=True)
thread.start()