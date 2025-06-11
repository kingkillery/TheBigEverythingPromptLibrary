// Basic search functionality for new UI
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function populateCategories() {
  const cats = await fetchJSON("/api/categories");
  const sel = document.getElementById("categoryFilter");
  Object.entries(cats).forEach(([cat, count]) => {
    const opt = document.createElement("option");
    opt.value = cat;
    opt.textContent = `${cat} (${count})`;
    sel.appendChild(opt);
  });
}

async function search() {
  const query = document.getElementById("searchInput").value;
  const category = document.getElementById("categoryFilter").value;
  const params = new URLSearchParams();
  if (query) params.append("query", query);
  if (category) params.append("category", category);
  const data = await fetchJSON(`/api/search?${params}`);

  document.getElementById("stats").textContent = `Found ${data.total} items`;
  const container = document.getElementById("results");
  container.innerHTML = "";
  data.items.forEach((item) => {
    const div = document.createElement("div");
    div.className = "p-4 bg-white rounded-md shadow cursor-pointer hover:bg-gray-50";
    div.innerHTML = `<h2 class='font-semibold text-lg text-blue-600'>${item.title}</h2>
      <p class='text-sm text-gray-500 mb-2'>${item.category}</p>
      <p>${item.description}</p>`;
    // Attach click handler to load full prompt
    div.addEventListener("click", () => showPrompt(item.id));
    container.appendChild(div);
  });
}

document.getElementById("searchInput").addEventListener("input", () => {
  clearTimeout(window.__searchT);
  window.__searchT = setTimeout(search, 300);
});
document.getElementById("categoryFilter").addEventListener("change", search);

(async function init() {
  await populateCategories();
  search();
})();

// NEW CODE: cache + modal display of prompt content
const promptCache = {};

async function showPrompt(itemId) {
  try {
    // Check cache first
    if (!promptCache[itemId]) {
      const data = await fetchJSON(`/api/item/${encodeURIComponent(itemId)}`);
      promptCache[itemId] = data;
    }
    const item = promptCache[itemId];
    renderPromptModal(item);
  } catch (err) {
    console.error("Failed to load item", err);
    alert("Failed to load prompt details. See console for more info.");
  }
}

function renderPromptModal(item) {
  // Remove any existing modal
  const existing = document.getElementById("promptModal");
  if (existing) existing.remove();

  // Create overlay
  const overlay = document.createElement("div");
  overlay.id = "promptModal";
  overlay.className = "fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50";

  // Modal content
  const modal = document.createElement("div");
  modal.className = "bg-white w-11/12 sm:w-3/4 lg:w-1/2 max-h-[80vh] overflow-y-auto p-6 rounded-md shadow-lg";

  // Title
  const title = document.createElement("h2");
  title.className = "text-xl font-bold mb-4";
  title.textContent = item.title || "Prompt";
  modal.appendChild(title);

  // Full content
  const pre = document.createElement("pre");
  pre.className = "whitespace-pre-wrap text-sm";
  pre.textContent = item.full_content || item.content || "No content available.";
  modal.appendChild(pre);

  // Close button
  const closeBtn = document.createElement("button");
  closeBtn.className = "mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700";
  closeBtn.textContent = "Close";
  closeBtn.addEventListener("click", () => overlay.remove());
  modal.appendChild(closeBtn);

  overlay.appendChild(modal);
  document.body.appendChild(overlay);
}

// Helper for POST requests returning JSON
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// Process prompt via /api/chat/process and render output
async function processPrompt() {
  const prompt = document.getElementById("chatPrompt").value.trim();
  if (!prompt) {
    alert("Please enter a prompt first.");
    return;
  }
  try {
    const data = await postJSON("/api/chat/process", { prompt, max_results: 5 });
    renderChatOutput(data);
  } catch (err) {
    console.error(err);
    alert("Failed to process prompt. See console for details.");
  }
}

function renderChatOutput(data) {
  const container = document.getElementById("chatOutput");
  container.innerHTML = "";

  // Optimized prompt
  if (data.optimized_prompt) {
    const optDiv = document.createElement("div");
    optDiv.className = "p-4 bg-white rounded-md shadow";
    optDiv.innerHTML = `<h3 class='font-semibold mb-2 text-green-700'>Optimized Prompt</h3><pre class='whitespace-pre-wrap text-sm'>${data.optimized_prompt}</pre>`;
    container.appendChild(optDiv);
  }

  // Tweaked top match
  if (data.tweaked_match) {
    const tweakDiv = document.createElement("div");
    tweakDiv.className = "p-4 bg-white rounded-md shadow";
    tweakDiv.innerHTML = `<h3 class='font-semibold mb-2 text-indigo-700'>Tweaked Top Match</h3><pre class='whitespace-pre-wrap text-sm'>${data.tweaked_match}</pre>`;
    container.appendChild(tweakDiv);
  }

  // Similar matches list
  if (data.matches && data.matches.length) {
    const matchesHeader = document.createElement("h3");
    matchesHeader.className = "font-semibold mt-4 mb-2";
    matchesHeader.textContent = "Similar Prompts";
    container.appendChild(matchesHeader);

    data.matches.forEach((item) => {
      const div = document.createElement("div");
      div.className = "p-4 bg-white rounded-md shadow cursor-pointer hover:bg-gray-50";
      div.innerHTML = `<h4 class='font-semibold text-blue-600'>${item.title}</h4>
        <p class='text-sm text-gray-500 mb-2'>${item.category}</p>
        <p>${item.description}</p>`;
      div.addEventListener("click", () => showPrompt(item.id));
      container.appendChild(div);
    });
  }
}

// Attach event listener after DOM ready
window.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("chatProcessBtn");
  if (btn) btn.addEventListener("click", processPrompt);
});
