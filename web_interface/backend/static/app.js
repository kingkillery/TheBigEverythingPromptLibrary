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
    div.className = "p-4 bg-white rounded-md shadow";
    div.innerHTML = `<h2 class='font-semibold text-lg text-blue-600'>${item.title}</h2>
      <p class='text-sm text-gray-500 mb-2'>${item.category}</p>
      <p>${item.description}</p>`;
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
