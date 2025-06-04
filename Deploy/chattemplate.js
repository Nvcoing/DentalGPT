// chattemplate.js

/**
 * Hàm lọc và format response thô từ API thành HTML chuẩn
 * theo yêu cầu của bạn:
 * - <｜think｜> block suy luận + <reasoning_cot> tạo mũi tên lên xuống và mờ phần suy luận
 * - <｜expert｜> block chuyên gia + <experting> tạo mũi tên và mờ
 * - <｜assistant｜> block trả lời + <answer> làm nổi bật câu trả lời
 * Cuối cùng trả về HTML chuỗi đã format.
 */
export function formatChatResponse(raw) {
  // Dùng regex để tách role block
  // Role start tags: <｜think｜>, <｜expert｜>, <｜assistant｜>
  // Role end tags không rõ, ta dựa vào thẻ con (reasoning_cot, experting, answer) để xử lý.

  // Bỏ ký tự unicode chữ "｜" (U+FF5C) thành ký tự pipe | để dễ xử lý
  raw = raw.replace(/<｜/g, "<|").replace(/｜>/g, "|>");

  // Tạo container để build HTML
  const container = document.createElement("div");

  // Helper tạo icon mũi tên lên xuống (toggle)
  const arrowIcon = () =>
    `<span class="inline-block transform transition-transform duration-300">&#x21C5;</span>`; // ↕ (up-down arrow)

  // Chia raw thành từng block theo role <|role|> ... <|role|>
  // Cách đơn giản là split theo dấu <| và parse từng block
  const blocks = raw.split(/<\|/).filter(Boolean);

  blocks.forEach((blockRaw) => {
    // blockRaw ví dụ: "think|> ... </reasoning_cot> ..."
    const matchRole = blockRaw.match(/^(\w+)\|>/);
    if (!matchRole) return;

    const role = matchRole[1];
    const content = blockRaw.replace(/^(\w+)\|>/, "").trim();

    if (role === "think") {
      // Tìm thẻ reasoning_cot và wrap + mờ phần reasoning, thêm mũi tên
      // Tách nội dung trong reasoning_cot
      const reasoningMatch = content.match(/<reasoning_cot>([\s\S]*?)<\/reasoning_cot>/);
      let reasoningHTML = "";
      if (reasoningMatch) {
        // Mờ reasoning, có toggle arrow trước
        reasoningHTML = `
          <div class="think-block border-l-4 border-blue-400 bg-blue-50 p-3 rounded-md">
            <div class="font-semibold text-blue-700 mb-1 flex items-center gap-2 cursor-pointer reasoning-toggle">
              ${arrowIcon()} <span>Suy luận của Bác sĩ</span>
            </div>
            <div class="reasoning-content text-gray-600 max-h-48 overflow-hidden transition-[max-height] duration-500">
              ${reasoningMatch[1].replace(/\n/g, "<br>")}
            </div>
          </div>
        `;
      } else {
        // Nếu không có reasoning_cot thì show toàn bộ content mờ
        reasoningHTML = `<div class="think-block bg-blue-50 p-3 rounded-md text-gray-600">${content.replace(/\n/g, "<br>")}</div>`;
      }
      container.insertAdjacentHTML("beforeend", reasoningHTML);
    } else if (role === "expert") {
      // Tương tự, tìm experting
      const expertMatch = content.match(/<experting>([\s\S]*?)<\/experting>/);
      let expertHTML = "";
      if (expertMatch) {
        expertHTML = `
          <div class="expert-block border-l-4 border-purple-500 bg-purple-50 p-3 rounded-md">
            <div class="font-semibold text-purple-700 mb-1 flex items-center gap-2 cursor-pointer expert-toggle">
              ${arrowIcon()} <span>Chuyên gia</span>
            </div>
            <div class="expert-content text-purple-700 max-h-36 overflow-hidden transition-[max-height] duration-500">
              ${expertMatch[1].replace(/\n/g, "<br>")}
            </div>
          </div>
        `;
      } else {
        expertHTML = `<div class="expert-block bg-purple-50 p-3 rounded-md text-purple-700">${content.replace(/\n/g, "<br>")}</div>`;
      }
      container.insertAdjacentHTML("beforeend", expertHTML);
    } else if (role === "assistant") {
      // Tìm answer, làm nổi bật
      const answerMatch = content.match(/<answer>([\s\S]*?)<\/answer>/);
      let answerHTML = "";
      if (answerMatch) {
        answerHTML = `
          <div class="assistant-block bg-green-50 border border-green-400 p-4 rounded-md text-green-900 whitespace-pre-wrap">
            ${answerMatch[1].replace(/\n/g, "<br>")}
          </div>
        `;
      } else {
        answerHTML = `<div class="assistant-block bg-green-50 p-4 rounded-md text-green-900">${content.replace(/\n/g, "<br>")}</div>`;
      }
      container.insertAdjacentHTML("beforeend", answerHTML);
    }
  });

  // Sau khi render, thêm event toggle cho reasoning và expert
  // Vì đây là DOM mới, nên gắn sự kiện sau khi trả về nội dung vào DOM ngoài
  return container.innerHTML;
}

/**
 * Hàm gọi sau khi gán nội dung HTML vào chatbox,
 * để thêm sự kiện toggle mờ nội dung reasoning/expert
 */
export function bindToggleEvents() {
  document.querySelectorAll(".reasoning-toggle").forEach((header) => {
    header.addEventListener("click", () => {
      const content = header.nextElementSibling;
      if (!content) return;
      if (content.style.maxHeight) {
        content.style.maxHeight = null;
        header.querySelector("span").style.transform = "rotate(0deg)";
      } else {
        content.style.maxHeight = content.scrollHeight + "px";
        header.querySelector("span").style.transform = "rotate(180deg)";
      }
    });
  });

  document.querySelectorAll(".expert-toggle").forEach((header) => {
    header.addEventListener("click", () => {
      const content = header.nextElementSibling;
      if (!content) return;
      if (content.style.maxHeight) {
        content.style.maxHeight = null;
        header.querySelector("span").style.transform = "rotate(0deg)";
      } else {
        content.style.maxHeight = content.scrollHeight + "px";
        header.querySelector("span").style.transform = "rotate(180deg)";
      }
    });
  });
}
