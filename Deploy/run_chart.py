from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64
from typing import Dict, Any
import sys
from contextlib import redirect_stdout, redirect_stderr

app = FastAPI(title="Chart Generator API", description="API to execute Python code and generate charts")

class TextRequest(BaseModel):
    content: str
    
class CodeExecutionResult:
    def __init__(self):
        self.charts = []
        self.outputs = []

def execute_python_code(code: str) -> tuple:
    """
    Thực thi code Python và trả về kết quả
    """
    # Tạo namespace riêng cho code execution
    namespace = {
        'plt': plt,
        'pd': pd,
        'np': np,
        'matplotlib': matplotlib
    }
    
    # Capture output
    output_buffer = io.StringIO()
    error_buffer = io.StringIO()
    chart_buffer = io.BytesIO()
    
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            exec(code, namespace)
        
        # Check if there's a plot to save
        if plt.get_fignums():  # Check if there are any figures
            plt.savefig(chart_buffer, format='png', bbox_inches='tight', dpi=150)
            chart_buffer.seek(0)
            chart_data = base64.b64encode(chart_buffer.getvalue()).decode()
            plt.close('all')
            return chart_data, output_buffer.getvalue()
        else:
            return None, output_buffer.getvalue()
            
    except Exception as e:
        error_msg = error_buffer.getvalue() or str(e)
        raise Exception(f"Code execution error: {error_msg}")
    finally:
        chart_buffer.close()

def extract_and_execute_code_blocks(content: str) -> str:
    """
    Tìm và thực thi các code block Python trong markdown
    """
    # Pattern để tìm code blocks Python
    pattern = r'```python\s*\[code biểu đồ\]\s*\n(.*?)\n```'
    
    def replace_code_block(match):
        code = match.group(1).strip()
        
        try:
            chart_data, output = execute_python_code(code)
            
            if chart_data:
                # Thay thế code block bằng image tag
                return f'<img src="data:image/png;base64,{chart_data}" alt="Generated Chart" style="max-width: 100%; height: auto;">'
            else:
                # Nếu không có biểu đồ, hiển thị output text
                return f'<pre><code>{output}</code></pre>' if output else '<p>Code executed successfully (no output)</p>'
                
        except Exception as e:
            return f'<div style="color: red; border: 1px solid red; padding: 10px; border-radius: 5px;">Error executing code: {str(e)}</div>'
    
    # Thay thế tất cả code blocks
    result = re.sub(pattern, replace_code_block, content, flags=re.DOTALL)
    
    return result

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Trang chủ với form test
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chart Generator</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
            textarea { width: 100%; height: 300px; font-family: monospace; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Chart Generator - Nha Khoa</h1>
        <form id="chartForm">
            <h3>Nhập nội dung markdown (có chứa code Python):</h3>
            <textarea id="content" placeholder="Nhập nội dung markdown với code blocks...">
# Thống Kê Nha Khoa

Dưới đây là biểu đồ thống kê số bệnh nhân theo tháng:

```python [code biểu đồ]
import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu thống kê nha khoa
months = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
patients = [120, 135, 98, 167, 154, 178]
treatments = [85, 92, 67, 123, 108, 134]

x = np.arange(len(months))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, patients, width, label='Số bệnh nhân', color='#2E86AB')
bars2 = ax.bar(x + width/2, treatments, width, label='Số ca điều trị', color='#A23B72')

ax.set_xlabel('Tháng')
ax.set_ylabel('Số lượng')
ax.set_title('Thống Kê Hoạt Động Nha Khoa 6 Tháng Đầu Năm')
ax.set_xticks(x)
ax.set_xticklabels(months)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Phân Tích Dịch Vụ

Biểu đồ tròn thể hiện tỷ lệ các dịch vụ:

```python [code biểu đồ]
import matplotlib.pyplot as plt

# Dữ liệu dịch vụ nha khoa
services = ['Khám tổng quát', 'Nhổ răng', 'Trám răng', 'Tẩy trắng', 'Niềng răng', 'Khác']
percentages = [30, 20, 25, 10, 10, 5]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']

plt.figure(figsize=(8, 8))
plt.pie(percentages, labels=services, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Tỷ Lệ Các Dịch Vụ Nha Khoa', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.show()
```
            </textarea>
            <br><br>
            <button type="submit">Tạo Biểu Đồ</button>
        </form>
        
        <div id="result" class="result" style="display: none;"></div>
        
        <script>
            document.getElementById('chartForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const content = document.getElementById('content').value;
                const resultDiv = document.getElementById('result');
                
                try {
                    const response = await fetch('/process', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ content: content })
                    });
                    
                    const result = await response.text();
                    resultDiv.innerHTML = result;
                    resultDiv.style.display = 'block';
                } catch (error) {
                    resultDiv.innerHTML = '<div style="color: red;">Error: ' + error.message + '</div>';
                    resultDiv.style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/process", response_class=HTMLResponse)
async def process_content(request: TextRequest):
    """
    Xử lý nội dung markdown và thực thi code
    """
    try:
        # Convert markdown to HTML and execute code
        processed_content = extract_and_execute_code_blocks(request.content)
        
        # Convert remaining markdown to basic HTML
        processed_content = processed_content.replace('\n# ', '\n<h1>').replace('# ', '<h1>')
        processed_content = processed_content.replace('\n## ', '\n<h2>').replace('## ', '<h2>')
        processed_content = processed_content.replace('\n### ', '\n<h3>').replace('### ', '<h3>')
        
        # Add closing tags for headers
        processed_content = re.sub(r'<h1>([^<\n]+)', r'<h1>\1</h1>', processed_content)
        processed_content = re.sub(r'<h2>([^<\n]+)', r'<h2>\1</h2>', processed_content)
        processed_content = re.sub(r'<h3>([^<\n]+)', r'<h3>\1</h3>', processed_content)
        
        # Convert newlines to HTML breaks
        processed_content = processed_content.replace('\n\n', '</p><p>').replace('\n', '<br>')
        processed_content = '<p>' + processed_content + '</p>'
        
        # Clean up extra p tags
        processed_content = processed_content.replace('<p></p>', '')
        processed_content = processed_content.replace('<p><h', '<h').replace('</h1></p>', '</h1>')
        processed_content = processed_content.replace('</h2></p>', '</h2>').replace('</h3></p>', '</h3>')
        
        return f"""
        <div style="font-family: Arial, sans-serif; line-height: 1.6; max-width: 100%;">
            {processed_content}
        </div>
        """
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/api/sample")
async def get_sample_data():
    """
    API endpoint trả về dữ liệu mẫu
    """
    return {
        "title": "Thống Kê Nha Khoa",
        "data": {
            "monthly_patients": [120, 135, 98, 167, 154, 178],
            "services": {
                "general_checkup": 30,
                "tooth_extraction": 20,
                "filling": 25,
                "whitening": 10,
                "braces": 10,
                "others": 5
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)