import asyncio
import threading
import torch
import gc
from transformers import TextStreamer

class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue, loop, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.queue = queue
        self.loop = loop

    def on_finalized_text(self, text: str, stream_end: bool = False):
        asyncio.run_coroutine_threadsafe(self.queue.put(text), self.loop)
        if stream_end:
            asyncio.run_coroutine_threadsafe(self.queue.put(None), self.loop)

def start_generation_thread(model, tokenizer, inputs, params, queue, loop):
    def generate():
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=params.get("max_new_tokens", 50),
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 0.9),
                top_k=params.get("top_k", 50),
                repetition_penalty=params.get("repetition_penalty", 1.0),
                do_sample=params.get("do_sample", True),
                streamer=CustomStreamer(tokenizer, queue, loop, skip_prompt=True),
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.empty_cache()
        gc.collect()

    threading.Thread(target=generate, daemon=True).start()
