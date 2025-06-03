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

nest_asyncio.apply()
app = FastAPI()

model_name = "NV9523/DentalGPT"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True
)
FastLanguageModel.for_inference(model)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

@app.post("/model/generate/")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    max_new_tokens = data.get("max_new_tokens", 50)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)
    top_k = data.get("top_k", 50)
    repetition_penalty = data.get("repetition_penalty", 1.0)
    do_sample = data.get("do_sample", True)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    response_queue = asyncio.Queue()

    class CustomStreamer(TextStreamer):
        def __init__(self, tokenizer, queue, **kwargs):
            super().__init__(tokenizer, **kwargs)
            self.queue = queue

        def on_finalized_text(self, text: str, stream_end: bool = False):
            asyncio.run_coroutine_threadsafe(self.queue.put(text), loop)
            if stream_end:
                asyncio.run_coroutine_threadsafe(self.queue.put(None), loop)

    loop = asyncio.get_event_loop()
    streamer = CustomStreamer(tokenizer, response_queue, skip_prompt=True)

    async def response_generator():
        while True:
            text = await response_queue.get()
            if text is None:
                break
            yield text
            await asyncio.sleep(0.01)

    def generate_in_thread():
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.empty_cache()
        gc.collect()

    threading.Thread(target=generate_in_thread, daemon=True).start()
    return StreamingResponse(response_generator(), media_type="text/markdown")

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

ngrok.set_auth_token("2trAEunRvTy9WfZNUVGRt4bMhpy_267Sj5MEej5a1A3pkfrhg")  # Thay bằng token thật
public_url = ngrok.connect(8000).public_url
print(f"Public URL: {public_url}")

thread = threading.Thread(target=run_api, daemon=True)
thread.start()
