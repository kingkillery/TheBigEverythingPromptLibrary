// Enhanced functionality for dopamine-inducing UI
let userStreak = parseInt(localStorage.getItem('userStreak') || '0');
let promptsViewed = parseInt(localStorage.getItem('promptsViewed') || '0');

async function fetchJSON(url, options = {}) {
  showSearchLoader(true);
  try {
    const res = await fetch(url, options);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  } finally {
    showSearchLoader(false);
  }
}

// Enhanced UI feedback functions
function showSearchLoader(show) {
  const loader = document.getElementById('searchLoader');
  if (loader) {
    loader.classList.toggle('hidden', !show);
  }
}

function updateProgressRing() {
  const ring = document.getElementById('progressRing');
  const streakDisplay = document.getElementById('streakCount');
  
  if (ring && streakDisplay) {
    const maxStreak = 10; // Maximum streak before reset
    const progress = (userStreak % maxStreak) / maxStreak;
    const circumference = 175.84; // 2 * π * 28
    const strokeDashoffset = circumference - (progress * circumference);
    
    ring.style.strokeDashoffset = strokeDashoffset;
    streakDisplay.textContent = userStreak;
    
    // Add celebration effect when streak increases
    if (userStreak > 0 && userStreak % 5 === 0) {
      createConfetti();
    }
  }
}

function incrementStreak() {
  userStreak++;
  localStorage.setItem('userStreak', userStreak.toString());
  updateProgressRing();
  showSuccessToast(`Streak: ${userStreak} 🔥`);
}

function incrementPromptsViewed() {
  promptsViewed++;
  localStorage.setItem('promptsViewed', promptsViewed.toString());
  if (promptsViewed % 3 === 0) {
    incrementStreak();
  }
}

// Create confetti effect
function createConfetti() {
  const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3'];
  
  for (let i = 0; i < 50; i++) {
    setTimeout(() => {
      const confetti = document.createElement('div');
      confetti.className = 'confetti';
      confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
      confetti.style.left = Math.random() * window.innerWidth + 'px';
      confetti.style.top = '-10px';
      confetti.style.borderRadius = Math.random() > 0.5 ? '50%' : '0';
      
      const animation = confetti.animate([
        { 
          transform: 'translateY(0) rotate(0deg)', 
          opacity: 1 
        },
        { 
          transform: `translateY(${window.innerHeight + 10}px) rotate(${360 + Math.random() * 360}deg)`, 
          opacity: 0 
        }
      ], {
        duration: 3000 + Math.random() * 2000,
        easing: 'cubic-bezier(0.25, 0.46, 0.45, 0.94)'
      });
      
      document.body.appendChild(confetti);
      animation.addEventListener('finish', () => confetti.remove());
    }, i * 50);
  }
}

// Enhanced toast with different types
function showSuccessToast(message, type = 'success') {
  const toast = document.createElement('div');
  const bgColor = type === 'success' ? 'bg-emerald-500' : type === 'error' ? 'bg-red-500' : 'bg-blue-500';
  
  toast.className = `fixed bottom-6 left-1/2 transform -translate-x-1/2 ${bgColor} text-white px-6 py-3 rounded-full shadow-lg z-50 text-sm font-medium animate-bounce-in`;
  toast.textContent = message;
  
  document.body.appendChild(toast);
  
  // Add sparkle effect for success
  if (type === 'success') {
    const sparkle = document.createElement('div');
    sparkle.innerHTML = '✨';
    sparkle.className = 'absolute -top-2 -right-2 animate-bounce';
    toast.appendChild(sparkle);
  }
  
  setTimeout(() => {
    toast.style.animation = 'fadeOut 0.3s ease-out forwards';
    setTimeout(() => toast.remove(), 300);
  }, 2000);
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

  // Enhanced color palette with more variety
  const colors = [
    "from-rose-400 to-fuchsia-500",
    "from-orange-400 to-amber-500", 
    "from-emerald-400 to-teal-500",
    "from-sky-400 to-indigo-500",
    "from-violet-400 to-purple-500",
    "from-lime-400 to-green-500",
    "from-pink-400 to-rose-500",
    "from-cyan-400 to-blue-500",
    "from-red-400 to-pink-500",
    "from-yellow-400 to-orange-500",
    "from-green-400 to-emerald-500",
    "from-blue-400 to-cyan-500",
  ];

  let i = 0;
  Object.entries(cats)
    .sort((a, b) => b[1] - a[1]) // larger categories first
    .forEach(([cat, count], index) => {
      const card = document.createElement("div");
      const color = colors[i % colors.length];
      i += 1;
      card.className = `group p-6 rounded-2xl shadow-lg border border-transparent cursor-pointer bg-gradient-to-br ${color} text-white hover:shadow-2xl card-interactive hover-grow`;
      card.style.animationDelay = `${index * 0.1}s`;
      card.classList.add('fade-in');
      
      card.innerHTML = `
        <div class="flex items-center justify-between mb-2">
          <h3 class="font-bold text-lg drop-shadow-sm">${cat}</h3>
          <div class="opacity-70 text-xl">🏷️</div>
        </div>
        <p class="text-sm opacity-90 font-medium">${count} prompts</p>
        <div class="mt-3 h-1 bg-white/20 rounded-full overflow-hidden">
          <div class="h-full bg-white/40 rounded-full transition-all duration-500" style="width: ${Math.min(count / 100 * 100, 100)}%"></div>
        </div>`;
        
      card.addEventListener("click", () => {
        // Add click feedback
        card.style.transform = 'scale(0.95)';
        setTimeout(() => {
          card.style.transform = '';
        }, 150);
        
        document.getElementById("categoryFilter").value = cat;
        // Smooth scroll to search section
        document.getElementById("searchInput").scrollIntoView({ 
          behavior: "smooth", 
          block: "center" 
        });
        
        // Add visual feedback
        showSuccessToast(`Exploring ${cat} prompts! 🌟`);
        search();
      });
      
      // Add hover sound effect (visual feedback instead of actual sound)
      card.addEventListener('mouseenter', () => {
        card.style.boxShadow = '0 25px 50px -12px rgba(0, 0, 0, 0.25)';
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
  
  // Show loading skeleton
  showLoadingSkeleton(true);
  
  try {
    const data = await fetchJSON(`/api/search?${params}`);

    // Update stats with animation
    const statsEl = document.getElementById("stats");
    statsEl.innerHTML = `
      <div class="flex items-center gap-2">
        <span class="text-2xl">🌸</span>
        <span class="font-bold text-lg">${data.total.toLocaleString()}</span>
        <span>prompts discovered</span>
        ${data.total > 0 ? '<span class="text-emerald-500">✨</span>' : ''}
      </div>`;
    
    const container = document.getElementById("results");
    container.innerHTML = "";
    
    // Show sort options if we have results
    document.getElementById("sortOptions").classList.toggle("hidden", data.total === 0);
    
    data.items.forEach((item, index) => {
      const div = document.createElement("div");
      div.className = "p-6 bg-white/80 backdrop-blur-sm border-2 border-gray-100 rounded-2xl shadow-lg hover:shadow-2xl hover:border-emerald-300 card-interactive cursor-pointer group";
      div.style.animationDelay = `${index * 0.1}s`;
      div.classList.add('fade-in');
      
      // Add category badge with color
      const categoryColors = {
        'Writing': 'bg-purple-100 text-purple-700',
        'Coding': 'bg-blue-100 text-blue-700', 
        'Business': 'bg-green-100 text-green-700',
        'Creative': 'bg-pink-100 text-pink-700',
        'Analysis': 'bg-orange-100 text-orange-700',
        'DeepResearch': 'bg-amber-100 text-amber-700',
        'default': 'bg-gray-100 text-gray-700'
      };
      
      const categoryColor = categoryColors[item.category] || categoryColors.default;
      
      div.innerHTML = `
        <div class="flex items-start justify-between mb-3">
          <h2 class='font-bold text-xl text-gray-800 group-hover:text-emerald-700 transition-colors duration-300 line-clamp-2'>${item.title}</h2>
          <div class="ml-3 flex-shrink-0">
            <span class='px-3 py-1 text-xs font-semibold rounded-full ${categoryColor}'>${item.category}</span>
          </div>
        </div>
        <p class='text-gray-600 mb-3 line-clamp-3'>${item.description}</p>
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2 text-sm text-gray-500">
            <span>📝</span>
            <span>Click to explore</span>
          </div>
          <div class="opacity-0 group-hover:opacity-100 transition-opacity duration-300">
            <span class="text-emerald-500 text-sm font-medium">→ View prompt</span>
          </div>
        </div>`;
        
      // Enhanced click handler with feedback
      div.addEventListener("click", () => {
        // Visual feedback on click
        div.style.transform = 'scale(0.98)';
        setTimeout(() => {
          div.style.transform = '';
        }, 100);
        
        incrementPromptsViewed();
        showPrompt(item.id);
      });
      
      container.appendChild(div);
    });
    
    // Hide loading skeleton
    showLoadingSkeleton(false);
    
    // Show celebration if this is a good search result
    if (data.total > 10) {
      setTimeout(() => {
        showSuccessToast(`Great search! Found ${data.total} perfect matches! 🎉`);
      }, 500);
    }
    
  } catch (error) {
    showLoadingSkeleton(false);
    showSuccessToast('Search failed. Please try again.', 'error');
  }
}

function showLoadingSkeleton(show) {
  const skeleton = document.getElementById('loadingSkeleton');
  const results = document.getElementById('results');
  
  if (skeleton && results) {
    skeleton.classList.toggle('hidden', !show);
    if (show) {
      results.innerHTML = '';
    }
  }
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

  // Enhanced Copy button with animation
  const copyBtn = document.createElement("button");
  copyBtn.className = "px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-xl hover:from-indigo-600 hover:to-purple-700 text-sm font-medium transform hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl flex items-center gap-2";
  copyBtn.innerHTML = `
    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
    </svg>
    <span>Copy</span>`;
  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(item.full_content || item.content || "");
      copyBtn.innerHTML = `
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
        </svg>
        <span>Copied!</span>`;
      copyBtn.className = copyBtn.className.replace('from-indigo-500 to-purple-600', 'from-green-500 to-emerald-600');
      
      // Create ripple effect
      createRippleEffect(copyBtn);
      showSuccessToast("Prompt copied to clipboard! ✨");
      
      setTimeout(() => {
        copyBtn.innerHTML = `
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
          </svg>
          <span>Copy</span>`;
        copyBtn.className = copyBtn.className.replace('from-green-500 to-emerald-600', 'from-indigo-500 to-purple-600');
      }, 2000);
    } catch {
      showSuccessToast("Failed to copy to clipboard", 'error');
    }
  });
  actions.appendChild(copyBtn);

  // Enhanced Close button
  const closeBtn = document.createElement("button");
  closeBtn.className = "px-6 py-3 bg-gradient-to-r from-gray-500 to-gray-600 text-white rounded-xl hover:from-gray-600 hover:to-gray-700 text-sm font-medium transform hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl";
  closeBtn.textContent = "Close";
  closeBtn.addEventListener("click", () => {
    overlay.style.animation = 'fadeOut 0.3s ease-out forwards';
    setTimeout(() => overlay.remove(), 300);
  });
  actions.appendChild(closeBtn);

  // Enhanced Add to Garden button
  const addBtn = document.createElement("button");
  addBtn.className = "px-6 py-3 bg-gradient-to-r from-emerald-500 to-green-600 text-white rounded-xl hover:from-emerald-600 hover:to-green-700 text-sm font-medium transform hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl flex items-center gap-2";
  addBtn.innerHTML = `
    <span>🌱</span>
    <span>Add to Garden</span>`;
  addBtn.addEventListener("click", async () => {
    const originalContent = addBtn.innerHTML;
    addBtn.innerHTML = `
      <div class="animate-spin h-4 w-4 border-2 border-white rounded-full border-t-transparent"></div>
      <span>Planting...</span>`;
    
    try {
      const userId = getUserId();
      // Fetch collections (simple, pick first)
      const colsRes = await fetch("/api/collections", { headers: { "X-User-Id": userId } });
      const cols = colsRes.ok ? await colsRes.json() : [];
      let target = cols[0];
      if (cols.length > 1) {
        const names = cols.map(c => `${c.id}: ${c.name}`).join("\n");
        const chosen = prompt(`Choose a garden bed to plant in (enter ID):\n${names}`, `${target.id}`);
        if (chosen) {
          const found = cols.find(c => c.id === Number(chosen));
          if (found) target = found;
        }
      }

      if (!target) {
        showSuccessToast("No garden found. Creating one for you!", 'error');
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
      
      addBtn.innerHTML = `
        <span>✅</span>
        <span>Planted!</span>`;
      addBtn.className = addBtn.className.replace('from-emerald-500 to-green-600', 'from-green-500 to-emerald-600');
      
      createConfetti();
      showSuccessToast("Prompt planted in your garden! 🌱✨");
      incrementStreak();
      
      setTimeout(() => {
        addBtn.innerHTML = originalContent;
        addBtn.className = addBtn.className.replace('from-green-500 to-emerald-600', 'from-emerald-500 to-green-600');
      }, 3000);
    } catch {
      addBtn.innerHTML = originalContent;
      showSuccessToast("Failed to add to garden", 'error');
    }
  });
  actions.appendChild(addBtn);

  // Enhanced Remix button
  const remixBtn = document.createElement("button");
  remixBtn.className = "px-6 py-3 bg-gradient-to-r from-fuchsia-500 to-pink-600 text-white rounded-xl hover:from-fuchsia-600 hover:to-pink-700 text-sm font-medium transform hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl flex items-center gap-2";
  remixBtn.innerHTML = `
    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
    </svg>
    <span>Remix</span>`;
  remixBtn.addEventListener("click", async () => {
    const originalContent = remixBtn.innerHTML;
    const style = prompt("Choose remix style: shorter | friendly | technical | creative", "shorter");
    if (!style) return;
    
    remixBtn.innerHTML = `
      <div class="animate-spin h-4 w-4 border-2 border-white rounded-full border-t-transparent"></div>
      <span>Remixing...</span>`;
    
    try {
      const res = await postJSON("/api/llm/enhance", {
        prompt: item.full_content || item.content || "",
        type: style,
        original_id: item.id,
      });
      
      remixBtn.innerHTML = originalContent;
      const result = res.enhanced_content || res.result || "No response";
      
      // Show enhanced result modal
      showRemixResult(result, style);
      showSuccessToast(`Prompt remixed in ${style} style! 🎨`);
      
    } catch (err) {
      remixBtn.innerHTML = originalContent;
      showSuccessToast("Failed to remix prompt", 'error');
    }
  });
  actions.appendChild(remixBtn);

// Create ripple effect for buttons
function createRippleEffect(button) {
  const ripple = document.createElement('div');
  ripple.className = 'absolute inset-0 rounded-xl';
  ripple.style.background = 'radial-gradient(circle, rgba(255,255,255,0.6) 0%, transparent 70%)';
  ripple.style.animation = 'ripple 0.6s ease-out';
  ripple.style.pointerEvents = 'none';
  
  const rect = button.getBoundingClientRect();
  button.style.position = 'relative';
  button.style.overflow = 'hidden';
  button.appendChild(ripple);
  
  setTimeout(() => ripple.remove(), 600);
}

function showRemixResult(result, style) {
  const modal = document.createElement('div');
  modal.className = 'fixed inset-0 bg-black/60 flex items-center justify-center z-60';
  modal.innerHTML = `
    <div class="bg-white rounded-2xl p-8 max-w-2xl max-h-[80vh] overflow-y-auto m-4 shadow-2xl">
      <h3 class="text-2xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-fuchsia-600 to-pink-600">
        ✨ Remixed Prompt (${style} style)
      </h3>
      <div class="bg-gray-50 rounded-xl p-4 mb-6">
        <pre class="whitespace-pre-wrap text-sm">${result}</pre>
      </div>
      <div class="flex justify-end gap-3">
        <button onclick="this.closest('.fixed').remove()" class="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors">
          Close
        </button>
        <button onclick="navigator.clipboard.writeText('${result.replace(/'/g, "\\'")}'); showSuccessToast('Remixed prompt copied! 🎉')" class="px-4 py-2 bg-fuchsia-500 text-white rounded-lg hover:bg-fuchsia-600 transition-colors">
          Copy Remix
        </button>
      </div>
    </div>`;
  document.body.appendChild(modal);
}

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

// Attach event listeners after DOM ready
window.addEventListener("DOMContentLoaded", () => {
  // Initialize progress ring
  updateProgressRing();
  
  const btn = document.getElementById("chatProcessBtn");
  if (btn) btn.addEventListener("click", processPrompt);
  
  const bedBtn = document.getElementById("myBedBtn");
  if (bedBtn) {
    bedBtn.addEventListener("click", showBedModal);
  } else {
    console.error("My Garden button not found!");
  }
  
  // Character counter for chat prompt
  const chatPrompt = document.getElementById("chatPrompt");
  const charCount = document.getElementById("charCount");
  if (chatPrompt && charCount) {
    chatPrompt.addEventListener("input", () => {
      const count = chatPrompt.value.length;
      charCount.textContent = count;
      charCount.className = count > 400 ? "text-red-500" : count > 300 ? "text-yellow-500" : "text-gray-400";
    });
  }
  
  // Enhanced search input with debouncing
  const searchInput = document.getElementById("searchInput");
  if (searchInput) {
    searchInput.addEventListener("focus", () => {
      searchInput.parentElement.classList.add("ring-4", "ring-emerald-200");
    });
    
    searchInput.addEventListener("blur", () => {
      searchInput.parentElement.classList.remove("ring-4", "ring-emerald-200");
    });
  }
  
  // Floating Action Button functionality
  const quickActionBtn = document.getElementById("quickActionBtn");
  if (quickActionBtn) {
    quickActionBtn.addEventListener("click", () => {
      // Quick scroll to search
      document.getElementById("searchInput").scrollIntoView({ 
        behavior: "smooth", 
        block: "center" 
      });
      setTimeout(() => {
        document.getElementById("searchInput").focus();
      }, 500);
    });
  }
  
  // Add keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    // Ctrl/Cmd + K to focus search
    if ((e.ctrlKey || e.metaKey) && e.key === "k") {
      e.preventDefault();
      document.getElementById("searchInput").focus();
      showSuccessToast("Search activated! 🔍");
    }
    
    // Escape to close modals
    if (e.key === "Escape") {
      const modals = document.querySelectorAll('[id$="Modal"]');
      modals.forEach(modal => modal.remove());
    }
  });
  
  // Welcome animation for first-time users
  if (!localStorage.getItem('hasVisited')) {
    setTimeout(() => {
      showSuccessToast("Welcome to Artificial Garden! 🌿✨");
      localStorage.setItem('hasVisited', 'true');
    }, 1000);
  }
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

// My Garden modal
function showBedModal() {
  const userId = getUserId();
  // UI boilerplate
  const overlay = document.createElement("div");
  overlay.id = "bedModal";
  overlay.className = "fixed inset-0 bg-black/40 flex items-center justify-center z-50";
  const box = document.createElement("div");
  box.className = "bg-white/90 backdrop-blur-md w-11/12 sm:w-3/4 lg:w-1/2 max-h-[80vh] overflow-y-auto p-6 rounded-xl shadow-2xl border border-emerald-200";
  box.innerHTML = `<h2 class="text-2xl font-bold mb-4 text-emerald-700">🌱 My Garden</h2>
    <div class="flex items-center gap-3 mb-4">
      <select id="bedSelect" class="border border-emerald-300 rounded px-3 py-1"></select>
      <button id="newBedBtn" class="px-3 py-1 bg-emerald-500 text-white rounded hover:bg-emerald-600 text-sm">New Bed</button>
      <button id="renameBedBtn" class="px-3 py-1 bg-amber-500 text-white rounded hover:bg-amber-600 text-sm">Rename</button>
      <button id="deleteBedBtn" class="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 text-sm">Delete</button>
    </div>
    <div id="bedContent" class="space-y-4"></div>`;
  const close = document.createElement("button");
  close.textContent = "Close";
  close.className = "mt-4 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 text-sm";
  close.onclick = () => overlay.remove();
  box.appendChild(close);
  overlay.appendChild(box);
  document.body.appendChild(overlay);

  // Load collections and populate select
  let collections = [];
  const bedSelect = () => document.getElementById("bedSelect");

  function loadBedItems(col) {
    if (!col) return;
    fetch(`/api/collections/${col.id}`, { headers: { "X-User-Id": userId } })
      .then(r => r.json())
      .then(data => {
        const content = document.getElementById("bedContent");
        content.innerHTML = "";
        if (!data.items.length) {
          content.innerHTML = "<p class='text-gray-600'>No prompts planted yet.</p>";
          return;
        }
        data.items.forEach(it => {
          const div = document.createElement("div");
          div.className = "p-3 bg-white border border-gray-200 rounded-md shadow flex justify-between items-center";
          div.innerHTML = `<div><h3 class='font-semibold'>${it.title}</h3><p class='text-xs text-gray-500'>${it.category}</p></div>`;
          const delBtn = document.createElement("button");
          delBtn.textContent = "🗑️";
          delBtn.className = "text-red-600 hover:text-red-800";
          delBtn.onclick = () => {
            fetch(`/api/collections/${col.id}/items/${encodeURIComponent(it.id)}`, {
              method: "DELETE",
              headers: { "X-User-Id": userId },
            }).then(() => {
              div.remove();
            });
          };
          div.appendChild(delBtn);
          div.addEventListener("click", () => showPrompt(it.id));
          content.appendChild(div);
        });
      });
  }

  function refreshCollectionSelect(selectedId) {
    const select = bedSelect();
    select.innerHTML = "";
    collections.forEach(c => {
      const opt = document.createElement("option");
      opt.value = c.id;
      opt.textContent = c.name;
      if (selectedId && Number(selectedId) === c.id) opt.selected = true;
      select.appendChild(opt);
    });
  }

  fetch(`/api/collections`, { headers: { "X-User-Id": userId } })
    .then(r => r.json())
    .then(cols => {
      collections = cols;
      if (!collections.length) {
        document.getElementById("bedContent").innerHTML = "<p class='text-gray-600'>Your garden is empty.</p>";
        return;
      }
      refreshCollectionSelect(collections[0].id);
      loadBedItems(collections[0]);
    });

  // Handlers
  box.querySelector("#bedSelect").addEventListener("change", e => {
    const col = collections.find(c => c.id === Number(e.target.value));
    loadBedItems(col);
  });

  box.querySelector("#newBedBtn").addEventListener("click", () => {
    const name = prompt("New garden bed name:", "New Bed");
    if (!name) return;
    fetch(`/api/collections`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-User-Id": userId },
      body: JSON.stringify({ name }),
    })
      .then(r => r.json())
      .then(col => {
        collections.unshift(col);
        refreshCollectionSelect(col.id);
        bedSelect().value = col.id;
        loadBedItems(col);
      });
  });

  box.querySelector("#renameBedBtn").addEventListener("click", () => {
    const current = collections.find(c => c.id === Number(bedSelect().value));
    const newName = prompt("Rename bed:", current.name);
    if (!newName || newName === current.name) return;
    fetch(`/api/collections/${current.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json", "X-User-Id": userId },
      body: JSON.stringify({ name: newName }),
    }).then(() => {
      current.name = newName;
      refreshCollectionSelect(current.id);
    });
  });

  box.querySelector("#deleteBedBtn").addEventListener("click", () => {
    const current = collections.find(c => c.id === Number(bedSelect().value));
    if (!current) return;
    if (!confirm(`Delete bed "${current.name}"? This cannot be undone.`)) return;
    fetch(`/api/collections/${current.id}`, {
      method: "DELETE",
      headers: { "X-User-Id": userId },
    }).then(() => {
      collections = collections.filter(c => c.id !== current.id);
      if (!collections.length) {
        document.getElementById("bedContent").innerHTML = "<p class='text-gray-600'>Your garden is empty.</p>";
        bedSelect().innerHTML = "";
      } else {
        refreshCollectionSelect(collections[0].id);
        loadBedItems(collections[0]);
      }
    });
  });
}
