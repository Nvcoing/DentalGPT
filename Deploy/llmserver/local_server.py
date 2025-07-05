import nest_asyncio
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from model_loader import load_model
from generator import start_generation_thread

nest_asyncio.apply()

app = FastAPI()
model, tokenizer, device = load_model()

@app.post("/model/generate/")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    params = {
        "max_new_tokens": data.get("max_new_tokens", 50),
        "temperature": data.get("temperature", 0.7),
        "top_p": data.get("top_p", 0.9),
        "top_k": data.get("top_k", 50),
        "repetition_penalty": data.get("repetition_penalty", 1.0),
        "do_sample": data.get("do_sample", True)
    }

    inputs = tokenizer(prompt, truncation=True, max_length=1024, return_tensors="pt").to(device)
    loop = asyncio.get_event_loop()
    response_queue = asyncio.Queue()

    start_generation_thread(model, tokenizer, inputs, params, response_queue, loop)

    async def response_generator():
        while True:
            text = await response_queue.get()
            if text is None:
                break
            yield text
            await asyncio.sleep(0.01)

    return StreamingResponse(response_generator(), media_type="text/markdown")

def run_local():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
