import mimetypes
import google.generativeai as genai
from gemini_tool.call_gemini import get_all_api_keys
api_keys = get_all_api_keys()
for key in api_keys:
    genai.configure(api_key=key)
def send_files_and_prompt(prompt, file_paths):
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    uploaded_files = []
    for path in file_paths:
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            # fallback cho pdf và ảnh phổ biến
            if path.lower().endswith('.pdf'):
                mime_type = 'application/pdf'
            elif path.lower().endswith(('.jpg', '.jpeg')):
                mime_type = 'image/jpeg'
            elif path.lower().endswith('.png'):
                mime_type = 'image/png'
            else:
                mime_type = 'application/octet-stream'
        uploaded_files.append(genai.upload_file(path, mime_type=mime_type))
    response = model.generate_content([prompt] + uploaded_files)
    return response.text
