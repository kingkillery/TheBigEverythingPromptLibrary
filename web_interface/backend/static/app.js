// Basic search functionality for new UI
async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
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
  renderCategoryGrid(cats);
}

// Render dynamic category grid cards
function renderCategoryGrid(cats) {
  const grid = document.getElementById("categoryGrid");
  if (!grid) return;
  grid.innerHTML = "";

  // Tailwind pastel background palette
  const colors = [
    "from-rose-400 to-fuchsia-500",
    "from-orange-400 to-amber-500",
    "from-emerald-400 to-teal-500",
    "from-sky-400 to-indigo-500",
    "from-violet-400 to-purple-500",
    "from-lime-400 to-green-500",
    "from-pink-400 to-rose-500",
    "from-cyan-400 to-blue-500",
  ];

  let i = 0;
  Object.entries(cats)
    .sort((a, b) => b[1] - a[1]) // larger categories first
    .forEach(([cat, count]) => {
      const card = document.createElement("div");
      const color = colors[i % colors.length];
      i += 1;
      card.className = `group p-4 rounded-xl shadow-lg border border-transparent cursor-pointer bg-gradient-to-br ${color} text-white hover:shadow-2xl transform hover:-translate-y-1 transition`;
      card.innerHTML = `
        <h3 class="font-semibold text-lg mb-1 drop-shadow-sm">${cat}</h3>
        <p class="text-sm opacity-90">${count} prompts</p>`;
      card.addEventListener("click", () => {
        document.getElementById("categoryFilter").value = cat;
        // Scroll to search section
        document.getElementById("searchInput").scrollIntoView({ behavior: "smooth", block: "center" });
        search();
      });
      grid.appendChild(card);
    });
}

async function search() {
  const query = document.getElementById("searchInput").value;
  const category = document.getElementById("categoryFilter").value;
  const params = new URLSearchParams();
  if (query) params.append("query", query);
  if (category) params.append("category", category);
  const data = await fetchJSON(`/api/search?${params}`);

  document.getElementById("stats").textContent = `🌸 ${data.total} blossoms in bloom`;
  const container = document.getElementById("results");
  container.innerHTML = "";
  data.items.forEach((item) => {
    const div = document.createElement("div");
    div.className = "p-4 bg-white/60 backdrop-blur-md border border-gray-200 rounded-xl shadow-md hover:shadow-xl hover:border-indigo-400 hover:-translate-y-1 transform transition duration-300 cursor-pointer";
    div.innerHTML = `<h2 class='font-semibold text-lg text-green-700'>${item.title}</h2>
      <p class='text-sm text-emerald-600 mb-2'>${item.category}</p>
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
  // Fetch total count quick via search API once
  try {
    const data = await fetchJSON("/api/search?limit=1");
    const total = data.total || 0;
    const pc = document.getElementById("promptCount");
    if (pc) pc.textContent = total.toLocaleString();
  } catch {}
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
  modal.className = "bg-white/90 backdrop-blur-lg w-11/12 sm:w-3/4 lg:w-1/2 max-h-[80vh] overflow-y-auto p-8 rounded-xl shadow-2xl border border-indigo-200";

  // Title
  const title = document.createElement("h2");
  title.className = "text-2xl font-extrabold bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 bg-clip-text text-transparent mb-4";
  title.textContent = item.title || "Prompt";
  modal.appendChild(title);

  // Usage guide placeholder
  const guideDiv = document.createElement("div");
  guideDiv.className = "bg-yellow-50 border border-yellow-200 p-3 rounded mb-4 text-sm text-yellow-800";
  guideDiv.textContent = "Loading guide...";
  modal.appendChild(guideDiv);

  const contentDiv = document.createElement("div");
  contentDiv.className = "prose max-w-none text-sm";
  contentDiv.innerHTML = formatMarkdown(item.full_content || item.content || "No content available.");
  modal.appendChild(contentDiv);

  // Fetch guide async
  (async () => {
    try {
      const g = await fetchJSON(`/api/guide/${encodeURIComponent(item.id)}`);
      guideDiv.innerHTML = `<strong>How to use:</strong> ${g.summary || ""}<br>${(g.advice || "").replace(/\n/g, "<br>")}`;
    } catch {
      guideDiv.remove();
    }
  })();

  // Action buttons container
  const actions = document.createElement("div");
  actions.className = "mt-6 flex justify-end gap-2";

  // Copy button
  const copyBtn = document.createElement("button");
  copyBtn.className = "px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 text-sm";
  copyBtn.textContent = "Copy";
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(item.full_content || item.content || "");
      copyBtn.textContent = "Copied!";
      setTimeout(() => (copyBtn.textContent = "Copy"), 1500);
    } catch {
      alert("Failed to copy to clipboard");
    }
  });
  actions.appendChild(copyBtn);

  // Close button
  const closeBtn = document.createElement("button");
  closeBtn.className = "px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm";
  closeBtn.textContent = "Close";
  closeBtn.addEventListener("click", () => overlay.remove());
  actions.appendChild(closeBtn);

  // Add to My Bed button
  const addBtn = document.createElement("button");
  addBtn.className = "px-4 py-2 bg-emerald-600 text-white rounded hover:bg-emerald-700 text-sm";
  addBtn.textContent = "Add to My Bed";
  addBtn.addEventListener("click", async () => {
    try {
      const userId = getUserId();
      // Fetch collections (simple, pick first)
      const colsRes = await fetch("/api/collections", { headers: { "X-User-Id": userId } });
      const cols = colsRes.ok ? await colsRes.json() : [];
      const target = cols[0];
      if (!target) {
        alert("No collection found.");
        return;
      }
      await fetch(`/api/collections/${target.id}/items`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": userId,
        },
        body: JSON.stringify({ prompt_id: item.id }),
      });
      addBtn.textContent = "Added!";
    } catch {
      alert("Failed to add to collection");
    }
  });
  actions.appendChild(addBtn);

  // Remix button
  const remixBtn = document.createElement("button");
  remixBtn.className = "px-4 py-2 bg-fuchsia-600 text-white rounded hover:bg-fuchsia-700 text-sm";
  remixBtn.textContent = "Remix";
  remixBtn.addEventListener("click", async () => {
    try {
      const style = prompt("Choose remix style: shorter | friendly | technical", "shorter");
      if (!style) return;
      remixBtn.textContent = "Remixing...";
      const res = await postJSON("/api/llm/enhance", {
        prompt: item.full_content || item.content || "",
        type: style,
        original_id: item.id,
      });
      remixBtn.textContent = "Remix";
      const result = res.enhanced_content || res.result || "No response";
      alert(result);
    } catch (err) {
      remixBtn.textContent = "Remix";
      alert("Failed to remix prompt");
    }
  });
  actions.appendChild(remixBtn);

  modal.appendChild(actions);
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

// Helper: Convert markdown to HTML (uses marked.js if available, otherwise fallback)
function formatMarkdown(md) {
  if (!md) return "";
  if (typeof window !== "undefined" && window.marked) {
    return window.marked.parse(md);
  }
  // Fallback: simple newline to <br> replacement
  return md.replace(/\n/g, "<br>");
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
    optDiv.className = "p-4 bg-white rounded-md shadow relative";
    optDiv.innerHTML = `<h3 class='font-semibold mb-2 text-green-700'>Optimized Prompt</h3>`;
    const pre = document.createElement("pre");
    pre.className = "whitespace-pre-wrap text-sm";
    pre.textContent = data.optimized_prompt;
    optDiv.appendChild(pre);
    // copy button
    const copyBtn = document.createElement("button");
    copyBtn.className = "absolute top-2 right-2 text-xs px-2 py-1 bg-gray-200 hover:bg-gray-300 rounded";
    copyBtn.textContent = "Copy";
    copyBtn.addEventListener("click", () => copyText(data.optimized_prompt));
    optDiv.appendChild(copyBtn);
    container.appendChild(optDiv);
  }

  // Tweaked top match
  if (data.tweaked_match) {
    const tweakDiv = document.createElement("div");
    tweakDiv.className = "p-4 bg-white rounded-md shadow relative";
    tweakDiv.innerHTML = `<h3 class='font-semibold mb-2 text-indigo-700'>Tweaked Top Match</h3>`;
    const pre2 = document.createElement("pre");
    pre2.className = "whitespace-pre-wrap text-sm";
    pre2.textContent = data.tweaked_match;
    tweakDiv.appendChild(pre2);
    const copyBtn2 = document.createElement("button");
    copyBtn2.className = "absolute top-2 right-2 text-xs px-2 py-1 bg-gray-200 hover:bg-gray-300 rounded";
    copyBtn2.textContent = "Copy";
    copyBtn2.addEventListener("click", () => copyText(data.tweaked_match));
    tweakDiv.appendChild(copyBtn2);
    container.appendChild(tweakDiv);
  }

  // Similar matches list
  if (data.matches && data.matches.length) {
    const matchesHeader = document.createElement("h3");
    matchesHeader.className = "font-semibold mt-4 mb-2";
    matchesHeader.textContent = "More blossoms to explore";
    container.appendChild(matchesHeader);

    data.matches.forEach((item) => {
      const div = document.createElement("div");
      div.className = "p-4 bg-white/60 backdrop-blur-md border border-gray-200 rounded-xl shadow-md hover:shadow-xl hover:border-indigo-400 hover:-translate-y-1 transform transition duration-300 cursor-pointer";
      div.innerHTML = `<h4 class='font-semibold text-blue-600'>${item.title}</h4>
        <p class='text-sm text-gray-500 mb-2'>${item.category}</p>
        <p>${item.description}</p>`;
      div.addEventListener("click", () => showPrompt(item.id));
      container.appendChild(div);
    });
  }
}

// Utility to copy text
function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(() => {
      toast("Copied!");
    });
  } else {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try {
      document.execCommand("copy");
      toast("Copied!");
    } catch {}
    ta.remove();
  }
}

// Simple toast
function toast(msg) {
  const t = document.createElement("div");
  t.textContent = msg;
  t.className = "fixed bottom-6 left-1/2 -translate-x-1/2 bg-black/80 text-white px-3 py-1 rounded-md z-50 text-sm";
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 1500);
}

// Attach event listener after DOM ready
window.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("chatProcessBtn");
  if (btn) btn.addEventListener("click", processPrompt);
});

// Utility: get or generate userId stored locally
function getUserId() {
  let uid = localStorage.getItem("gardenUserId");
  if (!uid) {
    uid = crypto.randomUUID();
    localStorage.setItem("gardenUserId", uid);
  }
  return uid;
}

async function ensureDefaultCollection() {
  const userId = getUserId();
  const res = await fetch("/api/collections", { headers: { "X-User-Id": userId } });
  const cols = res.ok ? await res.json() : [];
  if (!cols.length) {
    await fetch("/api/collections", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-User-Id": userId,
      },
      body: JSON.stringify({ name: "My Garden Bed" }),
    });
  }
}
// Call once on load to ensure user has at least one collection
ensureDefaultCollection().catch(() => {});
