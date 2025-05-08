from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pyngrok import ngrok
import random

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Mở tunnel public với ngrok
public_url = ngrok.connect(5000).public_url
print("* Ngrok tunnel URL:", public_url)

# Sample responses
sample_responses = [
    "Tôi có thể giúp gì cho vấn đề nha khoa của bạn?",
    "Để chẩn đoán chính xác hơn, bạn có thể mô tả triệu chứng chi tiết hơn không?",
    "Theo thông tin bạn cung cấp, có thể bạn đang gặp vấn đề về sâu răng. Bạn nên đến nha sĩ để kiểm tra cụ thể.",
    "Đối với cơn đau răng tạm thời, bạn có thể súc miệng bằng nước muối ấm và dùng thuốc giảm đau không kê đơn như paracetamol.",
    "Chi phí trám răng thường dao động từ 500.000đ đến 2.000.000đ tùy vào mức độ và loại vật liệu trám."
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    bot_response = random.choice(sample_responses)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    # Chạy Flask
    app.run()
