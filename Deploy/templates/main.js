const apiUrl = "http://localhost:8000/DentalGPT/chatbot/";
let isGenerating = false;
let currentBotMessage = null;
let streamBuffer = "";
const template = new ChatTemplate();
let sectionStates = {};

window.executeCode = async function(blockId) {
  const codeElement = document.getElementById(blockId);
  const outputElement = document.getElementById(blockId + '-output');
  const buttonElement = codeElement.parentElement.parentElement.querySelector('.run-button');
  const code = codeElement.textContent;
  buttonElement.disabled = true;
  buttonElement.textContent = "Đang chạy...";
  const loadingIndicator = buttonElement.parentElement.querySelector('.loading-indicator');
  loadingIndicator.style.display = 'inline';
  outputElement.style.display = 'none';
  try {
    const output = await template.executePythonCode(code);
    outputElement.innerHTML = output;
    outputElement.style.display = 'block';
  } catch (error) {
    outputElement.textContent = `Error: ${error.message}`;
    outputElement.style.display = 'block';
  } finally {
    buttonElement.disabled = false;
    buttonElement.textContent = "Chạy lại";
    loadingIndicator.style.display = 'none';
  }
};

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
  sectionStates = {};
}

function appendMessage(text, isUser) {
  const chatBox = document.getElementById("chat-box");
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

function bindCollapsibleEvents() {
  const chatBox = document.getElementById("chat-box");
  chatBox.removeEventListener('click', handleCollapsibleClick);
  chatBox.addEventListener('click', handleCollapsibleClick);
}

function handleCollapsibleClick(e) {
  const collapsible = e.target.closest('.collapsible');
  if (!collapsible) return;
  e.preventDefault();
  e.stopPropagation();
  const sectionKey = collapsible.dataset.section;
  const contentBox = collapsible.parentElement.querySelector('.content-box[data-content="' + sectionKey + '"]');
  const arrow = collapsible.querySelector('.arrow');
  if (contentBox && arrow) {
    const currentState = sectionStates[sectionKey] || false;
    const newState = !currentState;
    sectionStates[sectionKey] = newState;
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

async function sendMessage(mode = "normal") {
  if (isGenerating) return;
  const input = document.getElementById("user-input");
  const prompt = input.value.trim();
  if (!prompt) return;
  isGenerating = true;
  const sendBtn = document.getElementById("send-btn");
  sendBtn.textContent = "Đang xử lý...";
  sendBtn.disabled = true;
  appendMessage(prompt, true);
  input.value = "";
  currentBotMessage = appendMessage("", false);
  streamBuffer = "";
  sectionStates = {};
  const params = {
    prompt: prompt,
    max_new_tokens: parseInt(document.getElementById("max_tokens").value),
    temperature: parseFloat(document.getElementById("temperature").value),
    top_p: parseFloat(document.getElementById("top_p").value),
    top_k: 50,
    repetition_penalty: 1.0,
    do_sample: true,
    mode: mode
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
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      streamBuffer += chunk;
      const parsedContent = await parseStreamingResponse(streamBuffer);
      currentBotMessage.innerHTML = parsedContent + '<span class="stream-cursor"></span>';
      applySectionStates();
      const chatBox = document.getElementById("chat-box");
      chatBox.scrollTop = chatBox.scrollHeight;
      await new Promise(resolve => setTimeout(resolve, 30));
    }
    if (currentBotMessage) {
      const finalContent = await parseStreamingResponse(streamBuffer);
      currentBotMessage.innerHTML = finalContent;
      applySectionStates();
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

async function parseStreamingResponse(rawText) {
  let text = rawText.replace(/<｜[^｜]*｜>/g, "");
  const sections = {
    reasoning_cot: extractSection(text, 'reasoning_cot'),
    experting: extractSection(text, 'experting'),
    answer: extractSection(text, 'answer')
  };
  template.reset();
  Object.entries(sections).forEach(([key, content]) => {
    if (content) {
      template.updateSection(key, content);
    }
  });
  return await template.render();
}

document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('user-input').focus();
  bindCollapsibleEvents();
  initPyodide();
});