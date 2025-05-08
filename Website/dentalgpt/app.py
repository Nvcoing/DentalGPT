from flask import Flask, request, Response, render_template
from flask_cors import CORS
from pyngrok import ngrok
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
import torch
import threading
import markdown

app = Flask(__name__)
CORS(app)

# Khởi tạo Ngrok
public_url = ngrok.connect(5000).public_url
print("* Ngrok tunnel URL:", public_url)

# Load model từ Hugging Face qua Unsloth
model_name = "NV9523/DentalGPT_SFT"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    prompt = f"User: {user_message}\nBot:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Sử dụng TextIteratorStreamer để stream token
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def generate():
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            if " " in buffer:
                parts = buffer.split(" ")
                for word in parts[:-1]:
                    yield markdown.markdown(word + " ")  # stream mỗi từ sang HTML
                buffer = parts[-1]
        # yield từ cuối nếu còn
        if buffer:
            yield markdown.markdown(buffer)

    return Response(generate(), content_type='text/html; charset=utf-8')

if __name__ == '__main__':
    app.run()
