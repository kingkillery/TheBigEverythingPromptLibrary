'use strict';
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
    
    const promptIds = [];

    data.items.forEach((item, index) => {
      promptIds.push(item.id);
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
          <div class="flex items-center gap-3 text-sm text-gray-500">
            <span>💧 <span class="water-count" data-pid="${item.id}">0</span></span>
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
    
    // Fetch water/view counts in bulk
    if (promptIds.length) {
      try {
        const stats = await fetchJSON(`/api/usage-stats?ids=${promptIds.join(',')}`);
        Object.entries(stats).forEach(([pid, s]) => {
          const el = document.querySelector(`.water-count[data-pid="${pid}"]`);
          if (el) el.textContent = s.grafts ?? 0;
        });
      } catch (err) {
        console.warn('Could not load usage stats', err);
      }
    }
    
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
  await loadDiscoverySignals();
  // Fetch total count quick via search API once
  try {
    const data = await fetchJSON("/api/search?limit=1");
    const total = data.total || 0;
    const pc = document.getElementById("promptCount");
    if (pc) pc.textContent = total.toLocaleString();
  } catch {}
  search();
})();

// Load and display discovery signals (trending, popular, etc.)
async function loadDiscoverySignals() {
  try {
    const signals = await fetchJSON("/api/discovery-signals");
    renderDiscoverySignals(signals);
  } catch (error) {
    console.log("Could not load discovery signals:", error);
  }
}

function renderDiscoverySignals(signals) {
  // Create discovery signals section after category grid
  const categoryGrid = document.getElementById("categoryGrid");
  if (!categoryGrid) return;
  
  let discoverySection = document.getElementById("discoverySignals");
  if (!discoverySection) {
    discoverySection = document.createElement("div");
    discoverySection.id = "discoverySignals";
    discoverySection.className = "mt-12 mb-8";
    categoryGrid.parentNode.insertBefore(discoverySection, categoryGrid.nextSibling);
  }
  
  const signalTypes = [
    {
      key: 'trending_blossoms',
      title: '🔥 Trending Blossoms',
      description: 'Hot prompts gaining traction',
      color: 'from-red-400 to-orange-500'
    },
    {
      key: 'most_grafted', 
      title: '🌿 Most Grafted',
      description: 'Frequently remixed prompts',
      color: 'from-green-400 to-emerald-500'
    },
    {
      key: 'new_sprouts',
      title: '🌱 New Sprouts',
      description: 'Recently discovered gems',
      color: 'from-lime-400 to-green-500'
    },
    {
      key: 'popular_classics',
      title: '⭐ Popular Classics',
      description: 'Most viewed all-time',
      color: 'from-yellow-400 to-amber-500'
    }
  ];
  
  discoverySection.innerHTML = `
    <div class="text-center mb-8">
      <h2 class="text-3xl font-bold text-gray-800 mb-2">🌸 Garden Discovery</h2>
      <p class="text-gray-600">Explore what's blooming in the community</p>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      ${signalTypes.map(signalType => {
        const items = signals[signalType.key] || [];
        if (items.length === 0) return '';
        
        return `
          <div class="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden hover:shadow-xl transition-shadow duration-300">
            <div class="bg-gradient-to-r ${signalType.color} p-4 text-white">
              <h3 class="font-bold text-lg">${signalType.title}</h3>
              <p class="text-sm opacity-90">${signalType.description}</p>
            </div>
            <div class="p-4 space-y-3">
              ${items.slice(0, 3).map(item => `
                <div class="flex items-center justify-between text-sm cursor-pointer hover:bg-gray-50 p-2 rounded-lg transition-colors" onclick="showPrompt('${item.prompt.id}')">
                  <div class="flex-1 min-w-0">
                    <div class="font-medium text-gray-900 truncate">${item.prompt.title}</div>
                    <div class="text-gray-500 text-xs">${item.prompt.category}</div>
                  </div>
                  <div class="flex items-center gap-1 text-xs text-gray-400 ml-2">
                    ${item.views ? `<span title="Views">👁 ${item.views}</span>` : ''}
                    ${item.grafts ? `<span title="Grafts">🌿 ${item.grafts}</span>` : ''}
                    ${item.trending_score ? `<span title="Trending Score">🔥 ${Math.round(item.trending_score)}</span>` : ''}
                  </div>
                </div>
              `).join('')}
              
              ${items.length > 3 ? `
                <button class="w-full text-center text-sm text-gray-500 hover:text-gray-700 py-2 border-t border-gray-100 mt-3" onclick="showAllSignals('${signalType.key}', '${signalType.title}')">
                  +${items.length - 3} more ${signalType.title.toLowerCase()}
                </button>
              ` : ''}
            </div>
          </div>
        `;
      }).join('')}
    </div>
  `;
}

function showAllSignals(signalKey, title) {
  // Open modal showing all items for this signal type
  fetchJSON(`/api/${signalKey.replace('_', '-')}`).then(items => {
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black/60 flex items-center justify-center z-60';
    modal.innerHTML = `
      <div class="bg-white rounded-2xl p-8 max-w-4xl max-h-[80vh] overflow-y-auto m-4 shadow-2xl">
        <h2 class="text-2xl font-bold mb-6 text-gray-800">${title}</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          ${items.map(item => `
            <div class="p-4 border border-gray-200 rounded-xl hover:border-emerald-300 cursor-pointer transition-colors" onclick="showPrompt('${item.prompt.id}'); this.closest('.fixed').remove()">
              <h3 class="font-semibold text-gray-900 mb-1">${item.prompt.title}</h3>
              <p class="text-sm text-gray-600 mb-2">${item.prompt.description}</p>
              <div class="flex items-center justify-between text-xs text-gray-500">
                <span class="px-2 py-1 bg-gray-100 rounded">${item.prompt.category}</span>
                <div class="flex gap-2">
                  ${item.views ? `<span>👁 ${item.views}</span>` : ''}
                  ${item.grafts ? `<span>🌿 ${item.grafts}</span>` : ''}
                </div>
              </div>
            </div>
          `).join('')}
        </div>
        <div class="flex justify-center mt-6">
          <button onclick="this.closest('.fixed').remove()" class="px-6 py-3 bg-gray-500 text-white rounded-xl hover:bg-gray-600 transition-colors">
            Close
          </button>
        </div>
      </div>`;
    document.body.appendChild(modal);
  }).catch(() => {
    showSuccessToast("Could not load trending signals", 'error');
  });
}

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
    showSuccessToast("Failed to load prompt details. See console for more info.", 'error');
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

  // Enhanced usage guide section
  const guideDiv = document.createElement("div");
  guideDiv.className = "bg-gradient-to-r from-emerald-50 to-green-50 border border-emerald-200 p-4 rounded-xl mb-6 shadow-sm";
  guideDiv.innerHTML = `
    <div class="flex items-center gap-2 mb-3">
      <span class="text-lg">🌱</span>
      <h3 class="font-bold text-emerald-700">How to Use This Prompt</h3>
      <div class="animate-pulse h-2 w-2 bg-emerald-400 rounded-full"></div>
    </div>
    <div class="text-sm text-emerald-600">Loading personalized guidance...</div>`;
  modal.appendChild(guideDiv);

  const contentDiv = document.createElement("div");
  contentDiv.className = "prose max-w-none text-sm";
  contentDiv.innerHTML = formatMarkdown(item.full_content || item.content || "No content available.");
  modal.appendChild(contentDiv);

  // Fetch enhanced guide async
  (async () => {
    try {
      const g = await fetchJSON(`/api/guide/${encodeURIComponent(item.id)}`);
      
      let guideContent = `
        <div class="flex items-center gap-2 mb-3">
          <span class="text-lg">🌱</span>
          <h3 class="font-bold text-emerald-700">How to Use This Prompt</h3>
          ${g.related_guides_count > 0 ? `<span class="text-xs bg-emerald-100 text-emerald-600 px-2 py-1 rounded-full">${g.related_guides_count} guides referenced</span>` : ''}
        </div>`;
      
      if (g.summary) {
        guideContent += `<div class="mb-3 text-emerald-800 font-medium">${g.summary}</div>`;
      }
      
      if (g.tips) {
        guideContent += `
          <div class="mb-3">
            <h4 class="font-semibold text-emerald-700 mb-2 flex items-center gap-1">
              <span>💡</span> Key Tips
            </h4>
            <div class="text-sm text-emerald-600 space-y-1">
              ${g.tips.replace(/\n/g, "<br>")}
            </div>
          </div>`;
      }
      
      if (g.advice) {
        guideContent += `
          <div>
            <h4 class="font-semibold text-emerald-700 mb-2 flex items-center gap-1">
              <span>🎯</span> Best Practices
            </h4>
            <div class="text-sm text-emerald-600 space-y-1">
              ${g.advice.replace(/\n/g, "<br>")}
            </div>
          </div>`;
      }
      
      guideDiv.innerHTML = guideContent;
      
      // Add sparkle effect for enhanced guides
      if (g.related_guides_count > 0) {
        setTimeout(() => {
          const sparkle = document.createElement('div');
          sparkle.innerHTML = '✨';
          sparkle.className = 'absolute top-2 right-2 animate-bounce text-lg';
          guideDiv.style.position = 'relative';
          guideDiv.appendChild(sparkle);
          setTimeout(() => sparkle.remove(), 3000);
        }, 500);
      }
      
    } catch (error) {
      console.log("Could not load usage guide:", error);
      guideDiv.innerHTML = `
        <div class="flex items-center gap-2 mb-2">
          <span class="text-lg">🌱</span>
          <h3 class="font-bold text-emerald-700">Usage Guide</h3>
        </div>
        <div class="text-sm text-emerald-600">
          This is a ${item.category || 'general'} prompt. Use it to ${item.description || 'accomplish your task'}.
        </div>`;
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

  // Enhanced Remix button with preset graft options
  const remixBtn = document.createElement("button");
  remixBtn.className = "px-6 py-3 bg-gradient-to-r from-fuchsia-500 to-pink-600 text-white rounded-xl hover:from-fuchsia-600 hover:to-pink-700 text-sm font-medium transform hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl flex items-center gap-2";
  remixBtn.innerHTML = `
    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
    </svg>
    <span>Graft</span>`;
  remixBtn.addEventListener("click", () => {
    showGraftModal(item);
  });
  actions.appendChild(remixBtn);

  // Add Water button
  const waterBtn = document.createElement("button");
  waterBtn.className = "px-6 py-3 bg-gradient-to-r from-sky-500 to-blue-600 text-white rounded-xl hover:from-sky-600 hover:to-blue-700 text-sm font-medium transform hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl flex items-center gap-2";
  waterBtn.innerHTML = `<span>💧</span><span>Water</span>`;
  waterBtn.addEventListener("click", async () => {
    const original = waterBtn.innerHTML;
    waterBtn.innerHTML = `<div class='animate-spin h-4 w-4 border-2 border-white rounded-full border-t-transparent'></div><span>Watering...</span>`;
    try {
      const res = await fetch(`/api/prompts/${encodeURIComponent(item.id)}/water`, {
        method: "POST",
        headers: { "X-User-Id": getUserId() }
      });
      if (res.ok) {
        waterBtn.innerHTML = `<span>💧</span><span>Watered!</span>`;
        createConfetti();

        // Increment water count badge if present in list view
        const badge = document.querySelector(`.water-count[data-pid="${item.id}"]`);
        if (badge) {
          const current = parseInt(badge.textContent || '0', 10);
          badge.textContent = current + 1;
        }

        setTimeout(() => { waterBtn.innerHTML = original; }, 2000);
      } else {
        waterBtn.innerHTML = original;
        showSuccessToast("Failed to water ☹️", 'error');
      }
    } catch (e) {
      waterBtn.innerHTML = original;
    }
  });
  actions.appendChild(waterBtn);

  modal.appendChild(actions);
  overlay.appendChild(modal);
  document.body.appendChild(overlay);
}

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

// Enhanced Graft Modal with preset options
function showGraftModal(item) {
  const overlay = document.createElement('div');
  overlay.className = 'fixed inset-0 bg-black/60 flex items-center justify-center z-60';
  
  const graftOptions = [
    {
      type: 'shorter',
      title: '📝 Make it Concise',
      description: 'Trim the fat, keep the essence',
      icon: '✂️',
      color: 'from-blue-400 to-cyan-500'
    },
    {
      type: 'friendly',
      title: '😊 Make it Friendly',
      description: 'Warm, approachable, conversational',
      icon: '🤗',
      color: 'from-green-400 to-emerald-500'
    },
    {
      type: 'technical',
      title: '🔧 Make it Technical',
      description: 'Precise, detailed, expert-level',
      icon: '⚙️',
      color: 'from-purple-400 to-indigo-500'
    },
    {
      type: 'creative',
      title: '🎨 Make it Creative',
      description: 'Inspiring, imaginative, artistic',
      icon: '✨',
      color: 'from-pink-400 to-rose-500'
    },
    {
      type: 'expand',
      title: '📚 Expand & Detail',
      description: 'Add context, examples, depth',
      icon: '📈',
      color: 'from-amber-400 to-orange-500'
    },
    {
      type: 'variants',
      title: '🔄 Create Variants',
      description: '3 different approaches',
      icon: '🌟',
      color: 'from-violet-400 to-purple-500'
    }
  ];
  
  overlay.innerHTML = `
    <div class="bg-white/95 backdrop-blur-lg rounded-2xl p-8 max-w-4xl max-h-[90vh] overflow-y-auto m-4 shadow-2xl border border-purple-200">
      <div class="text-center mb-8">
        <h2 class="text-3xl font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-fuchsia-600 to-pink-600">
          🌿 Graft this Prompt
        </h2>
        <p class="text-gray-600">Choose how to transform and improve this prompt</p>
      </div>
      
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        ${graftOptions.map(option => `
          <button class="graft-option p-6 bg-gradient-to-br ${option.color} text-white rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition-all duration-300 text-left group" data-type="${option.type}">
            <div class="flex items-center gap-3 mb-3">
              <span class="text-2xl">${option.icon}</span>
              <span class="text-xl font-bold group-hover:text-yellow-200 transition-colors">${option.title}</span>
            </div>
            <p class="text-sm opacity-90 group-hover:opacity-100 transition-opacity">${option.description}</p>
            <div class="mt-4 flex justify-end opacity-0 group-hover:opacity-100 transition-opacity">
              <span class="text-xs bg-white/20 px-2 py-1 rounded-full">→ Transform</span>
            </div>
          </button>
        `).join('')}
      </div>
      
      <div class="flex justify-center">
        <button onclick="this.closest('.fixed').remove()" class="px-6 py-3 bg-gray-500 text-white rounded-xl hover:bg-gray-600 transition-colors font-medium">
          Cancel
        </button>
      </div>
    </div>`;
  
  // Add click handlers for graft options
  overlay.querySelectorAll('.graft-option').forEach(btn => {
    btn.addEventListener('click', async () => {
      const type = btn.dataset.type;
      const option = graftOptions.find(o => o.type === type);
      
      // Visual feedback
      btn.style.transform = 'scale(0.95)';
      btn.innerHTML = `
        <div class="flex items-center justify-center h-full">
          <div class="animate-spin h-8 w-8 border-4 border-white rounded-full border-t-transparent"></div>
          <span class="ml-3 font-medium">Grafting...</span>
        </div>`;
      
      try {
        const res = await postJSON("/api/llm/enhance", {
          prompt: item.full_content || item.content || "",
          type: type,
          original_id: item.id,
        });
        
        overlay.remove();
        const result = res.enhanced_content || res.result || "No response";
        showGraftResult(result, option);
        showSuccessToast(`${option.icon} Prompt grafted ${option.title.toLowerCase()}! 🌱`);
        
      } catch (err) {
        btn.style.transform = '';
        btn.innerHTML = `
          <div class="flex items-center gap-3 mb-3">
            <span class="text-2xl">${option.icon}</span>
            <span class="text-xl font-bold">${option.title}</span>
          </div>
          <p class="text-sm opacity-90">${option.description}</p>
          <p class="text-xs mt-2 bg-red-500/20 p-2 rounded">Failed to graft. Try again?</p>`;
        showSuccessToast("Grafting failed. Please try again.", 'error');
      }
    });
  });
  
  document.body.appendChild(overlay);
}

function showGraftResult(result, option) {
  const modal = document.createElement('div');
  modal.className = 'fixed inset-0 bg-black/60 flex items-center justify-center z-60';
  modal.innerHTML = `
    <div class="bg-white rounded-2xl p-8 max-w-4xl max-h-[80vh] overflow-y-auto m-4 shadow-2xl">
      <h3 class="text-3xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-fuchsia-600 to-pink-600">
        ${option.icon} ${option.title}
      </h3>
      <p class="text-gray-600 mb-6">${option.description}</p>
      
      <!-- Render area for grafted prompt -->
      <div id="graftResultContent" class="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-6 mb-6 border-2 border-purple-100 prose max-w-none text-sm leading-relaxed"></div>
      
      <div class="flex justify-end gap-3">
        <button onclick="this.closest('.fixed').remove()" class="px-6 py-3 bg-gray-500 text-white rounded-xl hover:bg-gray-600 transition-colors font-medium">
          Close
        </button>
        <button onclick="navigator.clipboard.writeText(\`${result.replace(/`/g, '\\`').replace(/\$/g, '\\$')}\`); showSuccessToast('Grafted prompt copied! ${option.icon}🎉')" class="px-6 py-3 bg-gradient-to-r from-fuchsia-500 to-pink-600 text-white rounded-xl hover:from-fuchsia-600 hover:to-pink-700 transition-all font-medium">
          Copy Grafted Prompt
        </button>
        <button onclick="showGraftModal({id: 'grafted', content: \`${result.replace(/`/g, '\\`').replace(/\$/g, '\\$')}\`, title: 'Grafted Prompt'})" class="px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl hover:from-green-600 hover:to-emerald-700 transition-all font-medium">
          Graft Again 🌿
        </button>
      </div>
    </div>`;

  // Inject formatted markdown into the result container
  const contentDiv = modal.querySelector('#graftResultContent');
  if (contentDiv) {
    contentDiv.innerHTML = formatMarkdown(result);
  }
  
  document.body.appendChild(modal);
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
  const promptElement = document.getElementById("chatPrompt");
  const prompt = promptElement.value.trim();
  if (!prompt) {
    showSuccessToast("Please enter a prompt first.", 'error');
    return;
  }

  const vineContainer = promptElement.closest(".vine-container");
  const btn = document.getElementById("chatProcessBtn");

  // Activate loading state
  vineContainer?.classList.add("loading");
  if (btn) btn.disabled = true;

  try {
    // Get selected prompt format
    const formatSelect = document.getElementById("promptFormat");
    const format = formatSelect ? formatSelect.value : "promptscript";
    const data = await postJSON("/api/chat/process", { prompt, max_results: 5, format });
    renderChatOutput(data);
  } catch (err) {
    console.error(err);
    showSuccessToast("Failed to process prompt. See console for details.", 'error');
  } finally {
    // Deactivate loading state
    vineContainer?.classList.remove("loading");
    if (btn) btn.disabled = false;
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

  const drBtn = document.getElementById("deepResearchBtn");
  if (drBtn) {
    drBtn.addEventListener("click", () => {
      document.getElementById("categoryFilter").value = "DeepResearch";
      document.getElementById("searchInput").scrollIntoView({ behavior: "smooth", block: "center" });
      showSuccessToast("Exploring Deep Research! 🧠");
      search();
    });
  }

  const plantBtn = document.getElementById("plantPromptBtn");
  const plantModal = document.getElementById("plantModal");
  if (plantBtn && plantModal) {
    const openModal = () => plantModal.classList.remove("hidden");
    const closeModal = () => plantModal.classList.add("hidden");
    plantBtn.addEventListener("click", openModal);
    document.getElementById("plantCancelBtn").addEventListener("click", closeModal);
    plantModal.addEventListener("click", (e) => { if (e.target === plantModal) closeModal(); });
    document.getElementById("plantSubmitBtn").addEventListener("click", async () => {
      const title = document.getElementById("plantTitle").value.trim();
      const promptText = document.getElementById("plantPromptText").value.trim();
      const category = document.getElementById("plantCategory").value.trim();
      const feedback = document.getElementById("plantFeedback");
      if (!title || !promptText) {
        feedback.textContent = "Title and prompt cannot be empty.";
        feedback.className = "text-red-600";
        return;
      }
      feedback.textContent = "Validating...";
      feedback.className = "text-gray-600";
      try {
        const res = await postJSON("/api/plant", {
          title,
          prompt: promptText,
          category_hint: category,
          user_id: getUserId(),
        });
        if (res.accepted) {
          feedback.innerHTML = `<span class='text-emerald-600'>🌸 Prompt planted successfully!</span>`;
          createConfetti();
          // auto-close modal after short delay
          setTimeout(closeModal, 1500);
        } else {
          feedback.innerHTML = `<span class='text-red-600'>Rejected at stage: ${res.stage_failed}</span>`;
        }
      } catch (err) {
        feedback.innerHTML = `<span class='text-red-600'>Error: ${err}</span>`;
      }
    });
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

// === AI News ===
async function loadAiNews() {
  try {
    const data = await fetchJSON('/api/ai-news?limit=15');
    renderAiNews(data.items || []);
  } catch (e) { console.log('AI news error', e); }
}

function renderAiNews(items) {
  const containerId = 'aiNewsPanel';
  let panel = document.getElementById(containerId);
  if (!panel) {
    panel = document.createElement('div');
    panel.id = containerId;
    panel.className = 'mt-10';
    const sidebar = document.getElementById('sidebar');
    if (sidebar) {
      sidebar.appendChild(panel);
    } else {
      // fallback: insert after discovery signals or at end of body
      const discovery = document.getElementById('discoverySignals');
      if (discovery && discovery.parentNode) {
        discovery.parentNode.insertBefore(panel, discovery.nextSibling);
      } else {
        document.body.appendChild(panel);
      }
    }
  }
  panel.innerHTML = `
    <h3 class="text-lg font-bold mb-3">📰 AI News</h3>
    <ul class="space-y-2 text-sm">
      ${items.slice(0,10).map(it => `<li><a class="text-blue-600 hover:underline" href="${it.link}" target="_blank">${it.title}</a></li>`).join('')}
    </ul>`;
}

// call on load
loadAiNews();

// ---------------- Self-trigger the DOMContentLoaded setup if the document has already loaded ----------------
if (document.readyState !== "loading") {
  // The script was loaded after DOMContentLoaded; fire the event handler manually so UI hooks attach.
  window.dispatchEvent(new Event("DOMContentLoaded"));
}
