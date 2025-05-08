from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pyngrok import ngrok
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Tạo ngrok public URL
public_url = ngrok.connect(5000).public_url
print("* Ngrok tunnel URL:", public_url)

# Tải mô hình nhẹ từ Hugging Face
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    # Tiền xử lý đầu vào
    prompt = f"User: {user_message}\nBot:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Sinh văn bản từ mô hình
    outputs = model.generate(
        inputs,
        max_length=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_response = generated_text.split("Bot:")[-1].strip()

    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run()
