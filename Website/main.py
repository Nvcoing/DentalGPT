import gradio as gr
from unsloth import FastLanguageModel
import torch
from transformers import TextIteratorStreamer
import threading

# Cấu hình
max_seq_length = 512
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=r"/content/drive/MyDrive/Project/DentalGPT_3",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)

system_prompt = (
    "Hãy là một chuyên gia về nha khoa đã có nhiều năm kinh nghiệm và có thể trả lời mọi câu hỏi một cách dễ hiểu "
    "cho người Việt Nam từ chuyên sâu đến đơn giản, dễ hiểu. Hãy trả lời câu hỏi dưới đây một cách đơn giản, đầy đủ, "
    "dễ hiểu, đúng trọng tâm và đúng ngữ cảnh bằng tiếng Việt:"
)

def build_prompt_cot(user_input, cot_goal, cot_reasoning, cot_justification):
    return (
        f"<|user|>\n{system_prompt}\n\n"
        f"Câu hỏi: {user_input}\n"
        f"Mục tiêu: {cot_goal}\n"
        f"Bước: {cot_reasoning}\n"
        f"Suy luận: {cot_justification}\n\n<|thought|>\n"
    )

def process_stream(stream_text):
    if "<|assistant|>" in stream_text:
        assistant_part = stream_text.split("<|assistant|>")[1].strip()
        if assistant_part.lower().startswith("câu trả lời:"):
            assistant_part = assistant_part[len("câu trả lời:"):].strip()
        return assistant_part
    return None

def chatbot_stream(user_input, cot_goal, cot_reasoning, cot_justification):
    prompt = build_prompt_cot(user_input, cot_goal, cot_reasoning, cot_justification)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        do_sample=True,
        temperature=0.4,
        top_p=0.85,
        repetition_penalty=1.25,
        pad_token_id=tokenizer.eos_token_id,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    assistant_response = ""
    for new_text in streamer:
        partial_text += new_text
        processed = process_stream(partial_text)
        if processed is not None:
            assistant_response = processed
            yield partial_text, assistant_response
        else:
            yield partial_text, ""

# ---------------- GIAO DIỆN GRADIO -------------------
with gr.Blocks(
    css="""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    body {
        background-color: #f5f7fa;
    }
    .chatbot-box {
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #4f46e5, #8b5cf6);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 12px;
    }
    .bot-message {
        background-color: white;
        border-radius: 18px 18px 18px 4px;
        padding: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .send-btn {
        background: linear-gradient(135deg, #4f46e5, #8b5cf6);
        color: white;
        border-radius: 9999px;
        padding: 8px 16px;
        font-weight: 600;
        transition: transform 0.2s ease;
    }
    .send-btn:hover {
        transform: scale(1.05);
    }
    """
) as demo:

    gr.HTML("""
    <header class="bg-white shadow-sm py-4 px-6 rounded-lg mb-4 flex items-center justify-between">
        <div class="flex items-center space-x-3">
            <div class="tooth-icon bg-indigo-600 w-10 h-10 rounded-lg flex items-center justify-center text-white">
                <i class="fas fa-tooth"></i>
            </div>
            <div>
                <h1 class="text-xl font-bold text-gray-800">DentalGPT</h1>
                <p class="text-xs text-gray-500">Trợ lý Nha khoa Thông minh</p>
            </div>
        </div>
    </header>
    """)

    chatbot = gr.Chatbot(label="", show_copy_button=True, elem_classes="chatbot-box", height=500)

    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(label="", placeholder="Nhập câu hỏi nha khoa tại đây...", lines=3)
        with gr.Column(scale=1):
            submit_btn = gr.Button("Gửi 🚀", elem_classes="send-btn")

    cot_goal = gr.Textbox(visible=False)
    cot_reasoning = gr.Textbox(visible=False)
    cot_justification = gr.Textbox(visible=False)

    with gr.Accordion("🧠 Chi tiết suy luận của AI", open=False):
        reasoning_output = gr.Textbox(label="Luồng suy luận", lines=8, interactive=False)
        final_output = gr.Textbox(label="Kết quả cuối cùng", lines=6, interactive=False)

    def respond(user_input, cot_goal, cot_reasoning, cot_justification, history):
        partial, answer = "", ""
        for partial, answer in chatbot_stream(user_input, cot_goal, cot_reasoning, cot_justification):
            yield history + [(user_input, answer)], partial, answer

    submit_btn.click(
        respond,
        inputs=[user_input, cot_goal, cot_reasoning, cot_justification, chatbot],
        outputs=[chatbot, reasoning_output, final_output]
    )

demo.launch(share=True)
