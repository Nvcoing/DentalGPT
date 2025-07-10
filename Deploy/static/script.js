const apiUrl = "http://localhost:8000/DentalGPT/chatbot/";
let isGenerating = false;
let currentBotMessage = null;
let streamBuffer = "";
let pyodide = null;
let isSearchEnabled = false;

function toggleSearch() {
  isSearchEnabled = !isSearchEnabled;
  const btn = document.getElementById('search-toggle-btn');
  if (isSearchEnabled) {
    btn.classList.remove('bg-yellow-500', 'hover:bg-yellow-600');
    btn.classList.add('bg-green-600', 'hover:bg-green-700');
    btn.textContent = '✅ Search ON';
  } else {
    btn.classList.remove('bg-green-600', 'hover:bg-green-700');
    btn.classList.add('bg-yellow-500', 'hover:bg-yellow-600');
    btn.textContent = '🔎 Search';
  }
}
// Initialize Pyodide with matplotlib support
async function initPyodide() {
  if (!pyodide) {
    try {
      pyodide = await loadPyodide();
      // Load required packages
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

// Auto-execute Python code and return result HTML
async function autoExecutePythonCode(code) {
  if (!pyodide) {
    await initPyodide();
  }
  
  if (!pyodide) {
    return `<div class="python-error">Lỗi: Không thể khởi tạo Python runner</div>`;
  }

  try {
    // Set up matplotlib backend and capture system
    await pyodide.runPythonAsync(`
      import matplotlib
      matplotlib.use('AGG')  # Use non-interactive backend
      import matplotlib.pyplot as plt
      import io
      import base64
      import sys
      from io import StringIO
      
      # Capture stdout for text output
      sys.stdout = StringIO()
      
      # Clear any existing plots
      plt.clf()
    `);
    
    // Run the user code
    await pyodide.runPythonAsync(code);
    
    // Get text output
    const textOutput = await pyodide.runPythonAsync("sys.stdout.getvalue()");
    
    // Try to get plot output
    let imageOutput = null;
    try {
      const hasPlot = await pyodide.runPythonAsync(`
        import matplotlib.pyplot as plt
        len(plt.get_fignums()) > 0
      `);
      
      if (hasPlot) {
        imageOutput = await pyodide.runPythonAsync(`
          import matplotlib.pyplot as plt
          import io
          import base64
          
          buf = io.BytesIO()
          plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
          buf.seek(0)
          encoded = base64.b64encode(buf.read()).decode('utf-8')
          plt.close()
          encoded
        `);
      }
    } catch (plotError) {
      console.log("No plot to save:", plotError);
    }
    
    // Build result HTML
    let resultHtml = '';
    
    // If there's a plot, show only the plot (replace the code)
    if (imageOutput) {
      resultHtml = `
        <div class="python-result">
          <div class="python-result-header">
            <span>📊</span>
            <span>Biểu đồ</span>
          </div>
          <div class="python-result-content">
            <div class="python-plot-output">
              <img src="data:image/png;base64,${imageOutput}" alt="Generated Plot" />
            </div>
          </div>
        </div>
      `;
    }
    // If there's text output but no plot, show the text
    else if (textOutput && textOutput.trim()) {
      resultHtml = `
        <div class="python-result">
          <div class="python-result-header">
            <span>🐍</span>
            <span>Kết quả Python</span>
          </div>
          <div class="python-result-content">
            <div class="python-text-output">${textOutput}</div>
          </div>
        </div>
      `;
    }
    // If code ran successfully but no output
    else {
      resultHtml = `
        <div class="python-result">
          <div class="python-result-header">
            <span>✅</span>
            <span>Python đã chạy thành công</span>
          </div>
          <div class="python-result-content">
            <div class="text-green-600 text-sm">Chương trình đã chạy thành công (không có output)</div>
          </div>
        </div>
      `;
    }
    
    return resultHtml;
    
  } catch (error) {
    return `
      <div class="python-result">
        <div class="python-result-header">
          <span>❌</span>
          <span>Lỗi Python</span>
        </div>
        <div class="python-result-content">
          <div class="python-error">Lỗi: ${error.message}</div>
        </div>
      </div>
    `;
  }
}

// Chat Template Class to manage rendering
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
    
    // Render collapsible sections with separate boxes
    for (const [key, section] of Object.entries(this.sections)) {
      if (key !== 'answer' && section.content.trim()) {
        html += await this.renderCollapsibleSection(key, section);
      }
    }

    // Render answer section
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
        <div class="flex items-center gap-2 mb3">
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
      // Process Python code blocks - auto-run and replace with results
      html = await this.processPythonCodeBlocks(html);
      return html;
    } catch (e) {
      return text.replace(/\n/g, '<br>');
    }
  }

  async processPythonCodeBlocks(html) {
    // Regex to find Python code blocks
    const pythonCodeRegex = /<pre><code class="language-python">([\s\S]*?)<\/code><\/pre>/g;
    // Process all matches
    const matches = [...html.matchAll(pythonCodeRegex)];
    for (const match of matches) {
      const fullMatch = match[0];
      const code = match[1].trim();
      const decodedCode = this.decodeHtml(code);

      // Show loading state first
      const loadingHtml = `
        <div class="python-execution-container">
          <div class="python-loading">
            <div class="spinner"></div>
            <span>Đang chạy code Python...</span>
          </div>
        </div>
      `;

      // Replace with loading first
      html = html.replace(fullMatch, loadingHtml);

      // Execute the code and get result
      let resultHtml = "";
      try {
        const resultHtml = await autoExecutePythonCode(decodedCode);
        // Replace loading with result
        html = html.replace(loadingHtml, resultHtml);
      } catch (error) {
        resultHtml = `
          <div class="python-result">
            <div class="python-result-header">
              <span>❌</span>
              <span>Lỗi Python</span>
            </div>
            <div class="python-result-content">
              <div class="python-error">Lỗi: ${error.message || "Không có biểu đồ hoặc output."}</div>
            </div>
          </div>
        `;
      }
      // Luôn thay thế loadingHtml bằng resultHtml (dù thành công hay lỗi)
      html = html.replace(loadingHtml, resultHtml);
    }
    return html;
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

const template = new ChatTemplate();

function handleKeyPress(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    if (!isGenerating) {
      sendMessage('normal');
    }
  }
}

function clearChat() {
  document.getElementById("chat-box").innerHTML = `
    <div class="text-gray-500 text-sm text-center py-8">
      <div class="text-2xl mb-2">👋</div>
      Xin chào! Tôi là DentalGPT. Hãy đặt câu hỏi về nha khoa...
    </div>
  `;
  // Reset section states
  sectionStates = {};
}

function appendMessage(text, isUser) {
  const chatBox = document.getElementById("chat-box");
  
  // Clear welcome message if exists
  if (chatBox.children.length === 1 && chatBox.children[0].classList.contains('text-center')) {
    chatBox.innerHTML = '';
  }

  const msg = document.createElement("div");
  msg.className = `message-container flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`;
  
  if (isUser) {
    msg.innerHTML = `
      <div class="bg-blue-600 text-white p-4 rounded-2xl max-w-[75%] shadow-lg">
        <div class="whitespace-pre-wrap">${escapeHtml(text)}</div>
      </div>
    `;
  } else {
    const botContent = document.createElement("div");
    botContent.className = "bg-gray-50 border border-gray-200 text-gray-900 p-4 rounded-2xl max-w-[90%] shadow-lg";
    
    if (text === "") {
      // Show typing indicator
      botContent.innerHTML = `
        <div class="flex items-center gap-2">
          <span class="typing-indicator"></span>
          <span class="typing-indicator"></span>
          <span class="typing-indicator"></span>
          <span class="text-sm text-gray-500 ml-2">Đang suy nghĩ...</span>
        </div>
      `;
    } else {
      botContent.innerHTML = text;
    }
    
    msg.appendChild(botContent);
    currentBotMessage = botContent;
  }
  
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
  
  return isUser ? null : currentBotMessage;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function extractSection(text, sectionName) {
  // Try to find complete section first
  const completeRegex = new RegExp(`<${sectionName}>([\\s\\S]*?)<\\/${sectionName}>`, 'g');
  const completeMatch = text.match(completeRegex);
  
  if (completeMatch) {
    return completeMatch[0].replace(new RegExp(`<\\/?${sectionName}>`, 'g'), '').trim();
  }
  
  // If no complete section, try to find incomplete/streaming section
  const incompleteRegex = new RegExp(`<${sectionName}>(([\\s\\S]*?)(?=<\\/|$))`, 'g');
  const incompleteMatch = text.match(incompleteRegex);
  
  if (incompleteMatch) {
    return incompleteMatch[0].replace(new RegExp(`<${sectionName}>`, 'g'), '').trim();
  }
  
  return "";
}

// Global state to track section visibility
let sectionStates = {};

function bindCollapsibleEvents() {
  // Use event delegation on the chat box to handle dynamic content
  const chatBox = document.getElementById("chat-box");
  
  // Remove existing listener if any
  chatBox.removeEventListener('click', handleCollapsibleClick);
  
  // Add event listener with delegation
  chatBox.addEventListener('click', handleCollapsibleClick);
}

function handleCollapsibleClick(e) {
  // Check if clicked element is a collapsible or its child
  const collapsible = e.target.closest('.collapsible');
  if (!collapsible) return;
  
  e.preventDefault();
  e.stopPropagation();
  
  const sectionKey = collapsible.dataset.section;
  const contentBox = collapsible.parentElement.querySelector('.content-box[data-content="' + sectionKey + '"]');
  const arrow = collapsible.querySelector('.arrow');
  
  if (contentBox && arrow) {
    // Toggle state
    const currentState = sectionStates[sectionKey] || false;
    const newState = !currentState;
    sectionStates[sectionKey] = newState;
    
    // Apply state
    if (newState) {
      contentBox.classList.add('visible');
      arrow.textContent = '▼';
      collapsible.classList.add('open');
    } else {
      contentBox.classList.remove('visible');
      arrow.textContent = '▶';
      collapsible.classList.remove('open');
    }
  }
}

function applySectionStates() {
  // Apply saved states to all sections
  Object.keys(sectionStates).forEach(sectionKey => {
    const isOpen = sectionStates[sectionKey];
    const collapsible = currentBotMessage ? currentBotMessage.querySelector(`.collapsible[data-section="${sectionKey}"]`) : null;
    const contentBox = currentBotMessage ? currentBotMessage.querySelector(`.content-box[data-content="${sectionKey}"]`) : null;
    
    if (collapsible && contentBox) {
      const arrow = collapsible.querySelector('.arrow');
      if (isOpen) {
        contentBox.classList.add('visible');
        arrow.textContent = '▼';
        collapsible.classList.add('open');
      } else {
        contentBox.classList.remove('visible');
        arrow.textContent = '▶';
        collapsible.classList.remove('open');
      }
    }
  });
}

async function parseStreamingResponse(rawText) {
  // Remove control tags
  let text = rawText.replace(/<｜[^｜]*｜>/g, "");
  
  // Extract sections using regex - improved for incomplete tags
  const sections = {
    reasoning_cot: extractSection(text, 'reasoning_cot'),
    experting: extractSection(text, 'experting'),
    answer: extractSection(text, 'answer')
  };

  // Update template
  template.reset();
  Object.entries(sections).forEach(([key, content]) => {
    if (content) {
      template.updateSection(key, content);
    }
  });

  return await template.render();
}

async function sendMessage(mode = "normal") {
  if (isGenerating) return;

  const input = document.getElementById("user-input");
  const prompt = input.value.trim();
  if (!prompt) return;

  const fileInput = document.getElementById("file-input");
  const files = fileInput.files;

  isGenerating = true;
  const sendBtn = document.getElementById("send-btn");
  sendBtn.textContent = "Đang xử lý...";
  sendBtn.disabled = true;

  appendMessage(prompt, true);
  input.value = "";
  currentBotMessage = appendMessage("", false);

  const params = {
    prompt: prompt,
    file_paths: fileInput.dataset.uploadedPaths ? JSON.parse(fileInput.dataset.uploadedPaths) : [],
    max_new_tokens: parseInt(document.getElementById("max_tokens").value),
    temperature: parseFloat(document.getElementById("temperature").value),
    top_p: parseFloat(document.getElementById("top_p").value),
    top_k: 50,
    repetition_penalty: 1.0,
    do_sample: true,
    mode: mode,
    module: isSearchEnabled ? "search_all" : null
  };

  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params)
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.statusText}`);
    }

    if (mode === "agentic") {
      // Clear typing indicator
      currentBotMessage.innerHTML = "";

      // Get JSON response
      const { response: agenticText } = await response.json();

      // Render agentic canvas
      const canvasHtml = `
        <div class="answer-section border-2 border-blue-600 bg-blue-50 p-4 rounded-xl shadow-lg">
          <div class="flex items-center gap-2 mb-3">
            <span class="text-lg">🔷</span>
            <span class="font-bold text-blue-800">Agentic Canvas</span>
          </div>
          <div class="markdown-content prose max-w-none">
            ${await template.markdownToHtml(agenticText)}
          </div>
        </div>
      `;
      currentBotMessage.innerHTML = canvasHtml;
      
    } else {
      // Handle streaming response for normal mode
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      streamBuffer = "";
      
      // Reset section states for new conversation
      sectionStates = {};

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        streamBuffer += chunk;
        
        // Parse and render the accumulated content
        const parsedContent = await parseStreamingResponse(streamBuffer);
        currentBotMessage.innerHTML = parsedContent + '<span class="stream-cursor"></span>';
        
        // Apply saved section states after content update
        applySectionStates();

        // Auto scroll
        const chatBox = document.getElementById("chat-box");
        chatBox.scrollTop = chatBox.scrollHeight;
        
        // Small delay for smooth streaming effect
        await new Promise(resolve => setTimeout(resolve, 30));
      }

      // Remove cursor and finalize
      if (currentBotMessage) {
        const finalContent = await parseStreamingResponse(streamBuffer);
        currentBotMessage.innerHTML = finalContent;
        applySectionStates();
      }
    }

  } catch (error) {
    if (currentBotMessage) {
      currentBotMessage.innerHTML = `
        <div class="text-red-600 flex items-center gap-2">
          <span>⚠️</span>
          <span>Đã xảy ra lỗi khi kết nối đến máy chủ: ${error.message}</span>
        </div>
      `;
    }
    console.error("Error:", error);
  } finally {
    isGenerating = false;
    sendBtn.textContent = "Gửi";
    sendBtn.disabled = false;
  }
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('user-input').focus();
  // Initialize event delegation for collapsibles
  bindCollapsibleEvents();
  // Initialize Pyodide in background
  initPyodide();
  
  // Toggle settings panel with animation
  const settingsBtn = document.getElementById('settings-btn');
  const settingsPanel = document.getElementById('settings-panel');
  
  settingsBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    settingsPanel.classList.toggle('hidden');
    if (!settingsPanel.classList.contains('hidden')) {
      settingsPanel.classList.remove('opacity-0', 'scale-95');
      settingsPanel.classList.add('opacity-100', 'scale-100');
    } else {
      settingsPanel.classList.remove('opacity-100', 'scale-100');
      settingsPanel.classList.add('opacity-0', 'scale-95');
    }
  });
  
  // Close settings when clicking outside
  document.addEventListener('click', (e) => {
    if (!settingsPanel.contains(e.target) && e.target !== settingsBtn) {
      settingsPanel.classList.add('hidden');
      settingsPanel.classList.remove('opacity-100', 'scale-100');
      settingsPanel.classList.add('opacity-0', 'scale-95');
    }
  });
});