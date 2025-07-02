import asyncio
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
