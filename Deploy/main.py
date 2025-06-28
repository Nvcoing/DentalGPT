from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from handlers.normal_handler import generate_response as normal_generate
from handlers.reason_handler import generate_response as reason_generate
from handlers.deep_reason_handler import generate_response as deep_reason_generate
from handlers.agentic_handler import generate_response as agentic_generate
from rag_search.internet_rag import Live_Retrieval_Augmented as rag_online 
from rag_search.vectordb_rag import run_keybert_qa as rag_local
from rag_search.google_search_api import tool_search as search
from rag_search.augmented import augmented as aug
from gemini_tool.read_file import send_files_and_prompt
from typing import List
import os
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.mount("/static", StaticFiles(directory="static"), name="static")
# Khai báo thư mục chứa templates (giao diện)
templates = Jinja2Templates(directory="templates")

# Route hiển thị giao diện chính
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/DentalGPT/upload_files/")
async def upload_files(
   files: List[UploadFile] = File(...)
):
    # Tạo thư mục nếu chưa có
    os.makedirs("temp_uploads", exist_ok=True)

    # Lưu file tạm thời
    file_paths = []
    for file in files:
        file_location = f"temp_uploads/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())
        file_paths.append(file_location)
    return {"file_paths": file_paths}
@app.post("/DentalGPT/chatbot/")
async def generate(request: Request):
    req_json = await request.json()
    prompt = req_json.get("prompt")
    file_paths = req_json.get("file_paths", [])  # <-- nhận danh sách file
    mode = req_json.get("mode", "normal")
    module = req_json.get("module", None)
    max_new_tokens = req_json.get("max_new_tokens", None)
    temperature = req_json.get("temperature", 0.1)
    top_p = req_json.get("top_p", 0.9)
    top_k = req_json.get("top_k", 50)
    repetition_penalty = req_json.get("repetition_penalty", 1.0)
    do_sample = req_json.get("do_sample", True)
    
    if not prompt:
        return JSONResponse(status_code=400, content={"error": "Missing 'prompt' in request"})
    if file_paths:
        result = send_files_and_prompt(prompt, file_paths)
        prompt = result
        for path in file_paths:
            os.remove(path)

    if module == "search_all":
        prompt = aug(prompt,rag_online(prompt,documents=search(prompt))["document"])
    elif module == "search_local":
        prompt = aug(prompt,rag_local(prompt,persist_dir="ChromaDB",top_k=5)["document"])
    
    # Gọi hàm tương ứng theo mode
    if mode == "reason":
        gen = reason_generate(prompt, temperature=temperature, top_p=top_p,
                              top_k=top_k, repetition_penalty=repetition_penalty,
                              do_sample=do_sample, max_new_tokens=max_new_tokens or 512)
        return StreamingResponse(gen, media_type="text/markdown")
    elif mode == "deep_reason":
        gen = deep_reason_generate(prompt, temperature=temperature, top_p=top_p,
                                   top_k=top_k, repetition_penalty=repetition_penalty,
                                   do_sample=do_sample, max_new_tokens=max_new_tokens or 512)
        return StreamingResponse(gen, media_type="text/markdown")
    elif mode == "agentic":
        gen = agentic_generate(prompt, temperature=temperature, top_p=top_p,
                                    top_k=top_k, repetition_penalty=repetition_penalty,
                                    do_sample=do_sample, max_new_tokens=max_new_tokens or 1024)
        output = ''.join(list(gen))
        full_output=output
        return JSONResponse(content={"response": full_output})
    else:
        gen = normal_generate(prompt, temperature=temperature, top_p=top_p,
                              top_k=top_k, repetition_penalty=repetition_penalty,
                              do_sample=do_sample, max_new_tokens=max_new_tokens or 256)
        return StreamingResponse(gen, media_type="text/markdown")


