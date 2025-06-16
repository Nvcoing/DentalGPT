let pyodide = null;

async function initPyodide() {
  if (!pyodide) {
    try {
      pyodide = await loadPyodide();
      await pyodide.loadPackage("micropip");
      await pyodide.runPythonAsync(`
        import micropip
        await micropip.install('matplotlib')
      `);
      console.log("Pyodide initialized successfully with matplotlib");
    } catch (error) {
      console.error("Failed to initialize Pyodide:", error);
    }
  }
  return pyodide;
}

class ChatTemplate {
  constructor() {
    this.sections = {
      reasoning_cot: { 
        title: "🧠 Quá trình suy luận", 
        content: "", 
        visible: false,
        badge: "SUY LUẬN",
        icon: "🤔"
      },
      experting: { 
        title: "👨‍🔬 Phân tích chuyên gia", 
        content: "", 
        visible: false,
        badge: "CHUYÊN GIA",
        icon: "🔬"
      },
      answer: { 
        title: "", 
        content: "", 
        visible: true 
      }
    };
  }

  reset() {
    Object.keys(this.sections).forEach(key => {
      this.sections[key].content = "";
      this.sections[key].visible = key === 'answer';
    });
  }

  updateSection(sectionName, content) {
    if (this.sections[sectionName]) {
      this.sections[sectionName].content = content;
    }
  }

  async render() {
    let html = "";
    for (const [key, section] of Object.entries(this.sections)) {
      if (key !== 'answer' && section.content.trim()) {
        html += await this.renderCollapsibleSection(key, section);
      }
    }
    if (this.sections.answer.content.trim()) {
      html += await this.renderAnswerSection(this.sections.answer.content);
    }
    return html;
  }

  async renderCollapsibleSection(key, section) {
    const extraClass = key === 'reasoning_cot' ? 'reasoning-box' : 'expert-box';
    const hasContent = section.content.trim() !== '';
    return `
      <div class="content-container">
        <div class="collapsible" data-section="${key}">
          <span class="arrow">▶</span>
          <span class="section-badge">${section.badge}</span>
          <span>${section.title}</span>
          ${hasContent ? '<span class="text-xs text-blue-500 ml-auto">Click để xem chi tiết</span>' : ''}
        </div>
        <div class="content-box ${extraClass}" data-content="${key}">
          <div class="markdown-content">${await this.markdownToHtml(section.content)}</div>
        </div>
      </div>
    `;
  }

  async renderAnswerSection(content) {
    return `
      <div class="answer-section">
        <div class="flex items-center gap-2 mb-3">
          <span class="text-lg">💡</span>
          <span class="font-semibold text-gray-700">Câu trả lời</span>
        </div>
        <div class="markdown-content prose max-w-none">
          ${await this.markdownToHtml(content)}
        </div>
      </div>
    `;
  }

  async markdownToHtml(text) {
    try {
      let html = marked.parse(text);
      html = await this.processPythonCodeBlocks(html);
      return html;
    } catch (e) {
      return text.replace(/\n/g, '<br>');
    }
  }

  async processPythonCodeBlocks(html) {
    const pythonCodeRegex = /<pre><code class="language-python">([\s\S]*?)<\/code><\/pre>/g;
    const matches = [...html.matchAll(pythonCodeRegex)];
    for (const match of matches) {
      const fullMatch = match[0];
      const code = match[1].trim();
      const decodedCode = this.decodeHtml(code);
      const blockId = 'python-' + Math.random().toString(36).substr(2, 9);
      let output = "";
      try {
        output = await this.executePythonCode(decodedCode);
      } catch (error) {
        output = `Error executing code: ${error.message}`;
      }
      const replacement = output
        ? `<div class="image-output"><img src="${output}" alt="Output Image" /></div>`
        : `
            <div class="python-code-block">
              <div class="python-code-header">
                <span>🐍 Python Code</span>
                <div>
                  <button class="run-button" onclick="executeCode('${blockId}')">Chạy lại</button>
                  <span class="loading-indicator">⏳</span>
                </div>
              </div>
              <div class="python-code-content">
                <pre id="${blockId}">${this.escapeHtml(decodedCode)}</pre>
              </div>
            </div>
          `;
      html = html.replace(fullMatch, replacement);
    }
    return html;
  }

  async executePythonCode(code) {
    if (!pyodide) {
      await initPyodide();
    }
    try {
      await pyodide.runPythonAsync(`
        import matplotlib
        matplotlib.use('AGG')
        import matplotlib.pyplot as plt
        import io
        import base64
      `);
      await pyodide.runPythonAsync(code);
      const imageBase64 = await pyodide.runPythonAsync(`
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        encoded
      `);
      return `data:image/png;base64,${imageBase64}`;
    } catch (error) {
      return `Error: ${error.message}`;
    }
  }

  decodeHtml(html) {
    const txt = document.createElement("textarea");
    txt.innerHTML = html;
    return txt.value;
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

function extractSection(text, sectionName) {
  const completeRegex = new RegExp(`<${sectionName}>([\\s\\S]*?)<\\/${sectionName}>`, 'g');
  const completeMatch = text.match(completeRegex);
  if (completeMatch) {
    return completeMatch[0].replace(new RegExp(`<\\/?${sectionName}>`, 'g'), '').trim();
  }
  const incompleteRegex = new RegExp(`<${sectionName}>(([\\s\\S]*?)(?=<\\/|$))`, 'g');
  const incompleteMatch = text.match(incompleteRegex);
  if (incompleteMatch) {
    return incompleteMatch[0].replace(new RegExp(`<${sectionName}>`, 'g'), '').trim();
  }
  return "";
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}