let chatHistory = [];

// DOM Elements
const initialInterface = document.getElementById('initialInterface');
const chatInterface = document.getElementById('chatInterface');
const chatContainer = document.getElementById('chatContainer');
const chatInput = document.getElementById('chatInput');

// Event Listeners
document.getElementById('initialSendButton').addEventListener('click', handleInitialSend);
document.getElementById('chatSendButton').addEventListener('click', handleChatSend);
chatInput.addEventListener('keypress', e => e.key === 'Enter' && handleChatSend());

// Handlers
function handleInitialSend() {
    const message = document.getElementById('initialInput').value.trim();
    if (message) {
        initialInterface.style.display = 'none';
        chatInterface.style.display = 'flex';
        addMessage(message, 'user');
        startChat(message);
    }
}

function handleChatSend() {
    const message = chatInput.value.trim();
    if (message) {
        chatInput.value = '';
        addMessage(message, 'user');
        startChat(message);
    }
}

async function startChat(message) {
    showTyping();
    
    try {
        const eventSource = new EventSource(`/chat?query=${encodeURIComponent(message)}`);
        let buffer = '';
        
        eventSource.onmessage = (e) => {
            buffer = e.data;
            updateLastBotMessage(buffer);
            scrollToBottom();
        };
        
        eventSource.onerror = () => {
            eventSource.close();
            hideTyping();
            addQuickQuestions();
            chatHistory.push({ role: 'assistant', content: buffer });
        };
        
    } catch (error) {
        console.error('Chat error:', error);
        addMessage('Xin lỗi, có lỗi xảy ra. Vui lòng thử lại.', 'bot');
        hideTyping();
    }
}

// UI Functions
function addMessage(content, role) {
    const div = document.createElement('div');
    div.className = `message ${role}-message`;
    div.textContent = content;
    chatContainer.appendChild(div);
    scrollToBottom();
}

function updateLastBotMessage(content) {
    const botMessages = document.querySelectorAll('.bot-message');
    if (botMessages.length > 0) {
        botMessages[botMessages.length - 1].textContent = content;
    }
}

function showTyping() {
    const typing = document.createElement('div');
    typing.className = 'message bot-message typing-indicator';
    typing.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    chatContainer.appendChild(typing);
    scrollToBottom();
}

function hideTyping() {
    document.querySelector('.typing-indicator')?.remove();
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addQuickQuestions() {
    const questions = [
        'Cách vệ sinh răng miệng?',
        'Chi phí niềng răng?',
        'Triệu chứng sâu răng?',
        'Có đau khi nhổ răng không?'
    ];
    
    const container = document.createElement('div');
    container.className = 'quick-questions';
    
    questions.forEach(q => {
        const btn = document.createElement('button');
        btn.className = 'quick-question';
        btn.textContent = q;
        btn.onclick = () => {
            container.remove();
            addMessage(q, 'user');
            startChat(q);
        };
        container.appendChild(btn);
    });
    
    chatContainer.appendChild(container);
    scrollToBottom();
}