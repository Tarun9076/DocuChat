const dropzone = document.getElementById("dropzone");
const dropzoneText = document.getElementById("dropzoneText");
const fileInput = document.getElementById("fileInput");
const docCard = document.getElementById("docCard");
const docName = document.getElementById("docName");
const docMeta = document.getElementById("docMeta");
const uploadStatus = document.getElementById("uploadStatus");

const messagesEl = document.getElementById("messages");
const emptyState = document.getElementById("emptyState");
const composer = document.getElementById("composer");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");

let documentLoaded = false;
let isStreaming = false;

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Minimal markdown: fenced code blocks, inline code, bold, paragraphs.
function renderMarkdown(raw) {
  const escaped = escapeHtml(raw);
  const withCodeBlocks = escaped.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
  const withInlineCode = withCodeBlocks.replace(/`([^`]+)`/g, "<code>$1</code>");
  const withBold = withInlineCode.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  const paragraphs = withBold
    .split(/\n{2,}/)
    .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
    .join("");
  return paragraphs;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addMessage(role, text) {
  emptyState.hidden = true;

  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "assistant" ? "AI" : "";

  const content = document.createElement("div");
  content.className = "msg-content";
  if (role === "user") {
    content.textContent = text;
  } else {
    content.innerHTML = renderMarkdown(text || "");
  }

  row.appendChild(avatar);
  row.appendChild(content);
  messagesEl.appendChild(row);
  scrollToBottom();
  return content;
}

function setComposerEnabled(enabled) {
  messageInput.disabled = !enabled || isStreaming;
  sendBtn.disabled = !enabled || isStreaming;
  messageInput.placeholder = enabled ? "Ask a question about the document…" : "Upload a document first…";
}

function autoGrow() {
  messageInput.style.height = "auto";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + "px";
}

messageInput.addEventListener("input", autoGrow);

messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

async function refreshStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  documentLoaded = data.document_loaded;
  if (documentLoaded) {
    docName.textContent = data.filename;
    docMeta.textContent = `${data.chunks} chunks indexed`;
    docCard.hidden = false;
    dropzoneText.textContent = "Replace document";
  }
  setComposerEnabled(documentLoaded);
}

async function uploadFile(file) {
  uploadStatus.textContent = `Indexing ${file.name}…`;
  uploadStatus.className = "upload-status";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      uploadStatus.textContent = data.error || "Upload failed.";
      uploadStatus.className = "upload-status error";
      return;
    }

    docName.textContent = data.filename;
    docMeta.textContent = `${data.chunks} chunks indexed`;
    docCard.hidden = false;
    dropzoneText.textContent = "Replace document";
    uploadStatus.textContent = "Ready.";
    uploadStatus.className = "upload-status success";

    documentLoaded = true;
    setComposerEnabled(true);
    messagesEl.innerHTML = "";
    emptyState.hidden = true;
    addMessage("assistant", `I've indexed **${data.filename}**. Ask me anything about it.`);
    messageInput.focus();
  } catch (err) {
    uploadStatus.textContent = "Upload failed: " + err.message;
    uploadStatus.className = "upload-status error";
  }
}

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) uploadFile(fileInput.files[0]);
});

["dragover", "dragenter"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

newChatBtn.addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  messagesEl.innerHTML = "";
  if (documentLoaded) {
    addMessage("assistant", "New chat started. Ask me anything about the document.");
  } else {
    messagesEl.appendChild(emptyState);
    emptyState.hidden = false;
  }
});

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = messageInput.value.trim();
  if (!text || isStreaming || !documentLoaded) return;

  addMessage("user", text);
  messageInput.value = "";
  autoGrow();

  isStreaming = true;
  setComposerEnabled(true);

  const assistantEl = addMessage("assistant", "");
  const cursor = document.createElement("span");
  cursor.className = "typing-cursor";
  assistantEl.appendChild(cursor);

  let fullText = "";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    if (!res.ok || !res.body) {
      const data = await res.json().catch(() => ({}));
      assistantEl.innerHTML = renderMarkdown(data.error || "Something went wrong.");
    } else {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        fullText += decoder.decode(value, { stream: true });
        assistantEl.innerHTML = renderMarkdown(fullText);
        assistantEl.appendChild(cursor);
        scrollToBottom();
      }
      assistantEl.innerHTML = renderMarkdown(fullText);
    }
  } catch (err) {
    assistantEl.innerHTML = renderMarkdown("Error: " + err.message);
  }

  isStreaming = false;
  setComposerEnabled(documentLoaded);
  messageInput.focus();
});

refreshStatus();
