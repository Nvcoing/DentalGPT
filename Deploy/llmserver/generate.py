from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import asyncio, threading, gc, torch

from model import model, tokenizer, device
from streamer import CustomStreamer

router = APIRouter()

@router.post("/model/generate/")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    max_new_tokens = data.get("max_new_tokens", 50)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 0.9)
    top_k = data.get("top_k", 50)
    repetition_penalty = data.get("repetition_penalty", 1.0)
    do_sample = data.get("do_sample", True)

    inputs = tokenizer(prompt, truncation=True, max_length=1024, return_tensors="pt").to(device)
    response_queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    streamer = CustomStreamer(tokenizer, response_queue, loop, skip_prompt=True)

    async def response_generator():
        while True:
            text = await response_queue.get()
            if text is None:
                break
            yield text
            await asyncio.sleep(0.01)

    def generate_text():
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

    threading.Thread(target=generate_text, daemon=True).start()
    return StreamingResponse(response_generator(), media_type="text/markdown")
