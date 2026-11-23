const categories = [
  { key: 'microscopy', label: 'Microscopía', detail: 'Fotografía convencional' },
  { key: 'collage', label: 'Collage', detail: 'Composición de imágenes' },
  { key: 'slide', label: 'Diapositiva', detail: 'Página con texto y fotos' },
  { key: 'illustration', label: 'Ilustración', detail: 'Dibujo o esquema digital' },
  { key: 'uncertain', label: 'Dudosa', detail: 'Revisar más tarde' },
];
const people = ['Alejandro', 'Rodrigo', 'Lucas'];
const state = { images: [], batches: {}, labels: {}, rater: null, filter: 'all', query: '', selected: new Set(), lastClicked: null, rendered: 0, undo: [], quickIndex: 0, quickOrder: [], classifying: false };
const $ = (id) => document.getElementById(id);
const gallery = $('gallery');
const observer = new IntersectionObserver((entries) => entries.forEach((entry) => {
  if (entry.isIntersecting) { const img = entry.target; img.src = img.dataset.src; observer.unobserve(img); }
}), { rootMargin: '240px' });

async function init() {
  const [imagesResponse, labelsResponse] = await Promise.all([fetch('/api/images'), fetch('/api/labels')]);
  const imageData = await imagesResponse.json();
  state.images = imageData.images;
  state.batches = imageData.batches;
  state.labels = (await labelsResponse.json()).labels;
  buildPeopleChoices(); buildBulkChoices();
  $('welcomeDialog').showModal();
}
function assignedImages(person = state.rater) {
  if (!person) return [];
  const assigned = new Set(state.batches[person] || []);
  return state.images.filter((image) => assigned.has(image.id));
}
function buildPeopleChoices() {
  $('peopleChoices').innerHTML = people.map((person, index) => {
    const total = (state.batches[person] || []).length;
    return `<button class="person-choice" data-person="${person}"><span class="person-avatar">${person[0]}</span><span class="person-choice-copy"><b>${person}</b><small>${total} imágenes asignadas</small></span><span class="person-arrow">→</span></button>`;
  }).join('');
  $('peopleChoices').querySelectorAll('[data-person]').forEach((button) => button.addEventListener('click', () => startAs(button.dataset.person)));
}
function startAs(person) {
  state.rater = person; state.filter = 'pending'; state.query = ''; $('search').value = '';
  $('personName').textContent = person; $('personBar').hidden = false;
  if ($('welcomeDialog').open) $('welcomeDialog').close();
  buildFilters(); updateStats(); renderGallery(); openQuick('pending');
}
function currentImages() {
  return assignedImages().filter((image) => {
    const category = state.labels[image.id];
    const inFilter = state.filter === 'all' || (state.filter === 'pending' ? !category : category === state.filter);
    return inFilter && (!state.query || image.id.includes(state.query));
  });
}
function buildFilters() {
  const mine = assignedImages();
  const counts = Object.fromEntries(categories.map((c) => [c.key, mine.filter((image) => state.labels[image.id] === c.key).length]));
  const classified = mine.filter((image) => state.labels[image.id]).length;
  const items = [['all', 'Todas', mine.length], ['pending', 'Pendientes', mine.length - classified], ...categories.map((c) => [c.key, c.label, counts[c.key]])];
  $('filters').innerHTML = items.map(([key, title, count]) => `<button class="filter ${state.filter === key ? 'active' : ''}" data-filter="${key}">${title}<span class="count">${count}</span></button>`).join('');
  $('filters').querySelectorAll('[data-filter]').forEach((button) => button.addEventListener('click', () => { state.filter = button.dataset.filter; state.selected.clear(); buildFilters(); renderGallery(); updateBulk(); }));
}
function buildBulkChoices() {
  $('bulkCategories').innerHTML = categories.map((c, i) => `<button class="category-button" data-category="${c.key}">${i + 1} · ${c.label}</button>`).join('');
  $('bulkCategories').querySelectorAll('[data-category]').forEach((button) => button.addEventListener('click', () => assign([...state.selected], button.dataset.category)));
  $('quickChoices').innerHTML = categories.map((c, i) => `<button class="choice" data-category="${c.key}"><kbd>${i + 1}</kbd>${c.label}<small>${c.detail}</small></button>`).join('');
  $('quickChoices').querySelectorAll('[data-category]').forEach((button) => button.addEventListener('click', () => classifyQuick(button.dataset.category)));
}
function updateStats() {
  const mine = assignedImages(), done = mine.filter((image) => state.labels[image.id]).length, pending = mine.length - done;
  $('progressText').textContent = `${state.rater ? `${state.rater}: ` : ''}${done} / ${mine.length} clasificadas · ${pending} pendientes`;
  $('progressBar').style.width = `${mine.length ? done / mine.length * 100 : 0}%`;
  $('stats').innerHTML = `<div class="stat"><i class="stat-dot"></i><span class="stat-num">${done}</span><span class="stat-label">Clasificadas</span></div><div class="stat"><i class="stat-dot" style="background:#647169"></i><span class="stat-num">${pending}</span><span class="stat-label">Pendientes</span></div>${categories.map((c) => `<div class="stat"><i class="stat-dot" style="background:${categoryColor(c.key)}"></i><span class="stat-num">${mine.filter((image) => state.labels[image.id] === c.key).length}</span><span class="stat-label">${c.label}</span></div>`).join('')}`;
  buildFilters();
}
function categoryColor(key) { return ({ microscopy: '#a3e635', collage: '#75c9ee', slide: '#ce9cff', illustration: '#ffc16d', uncertain: '#f28d80' })[key]; }
function renderGallery() {
  gallery.innerHTML = ''; state.rendered = 0;
  const images = currentImages(); $('emptyState').hidden = images.length !== 0;
  $('visibleCount').textContent = `${images.length.toLocaleString('es')} imágenes en el filtro`;
  appendNext(images, 72);
  $('sentinel').hidden = state.rendered >= images.length;
  if (window.galleryObserver) window.galleryObserver.disconnect();
  window.galleryObserver = new IntersectionObserver((entries) => { if (entries.some((e) => e.isIntersecting)) appendNext(images, 72); }, { rootMargin: '500px' });
  window.galleryObserver.observe($('sentinel'));
}
function appendNext(images, amount) {
  const fragment = document.createDocumentFragment();
  const end = Math.min(state.rendered + amount, images.length);
  for (; state.rendered < end; state.rendered++) {
    const image = images[state.rendered], category = state.labels[image.id];
    const card = document.createElement('article'); card.className = `card ${state.selected.has(image.id) ? 'selected' : ''}`; card.dataset.id = image.id;
    card.innerHTML = `<div class="thumb-wrap"><img class="thumb" data-src="/image/${image.id}" alt="Imagen ${image.id}" loading="lazy"><span class="select-check">${state.selected.has(image.id) ? '✓' : ''}</span><button class="open-image" title="Ampliar imagen" aria-label="Ampliar">⤢</button></div><div class="card-info"><span class="image-id">${image.id}</span>${category ? `<span class="badge ${category}">${categories.find((c) => c.key === category).label}</span>` : ''}</div>`;
    card.addEventListener('click', (event) => { if (event.target.closest('.open-image')) { event.stopPropagation(); openLightbox(image); return; } selectCard(image.id, event.shiftKey); });
    card.querySelector('.open-image').addEventListener('click', (event) => { event.stopPropagation(); openLightbox(image); });
    fragment.append(card);
  }
  gallery.append(fragment); gallery.querySelectorAll('img[data-src]').forEach((img) => observer.observe(img));
  $('sentinel').hidden = state.rendered >= images.length;
}
function selectCard(id, range) {
  if (range && state.lastClicked) {
    const list = currentImages().map((image) => image.id), a = list.indexOf(state.lastClicked), b = list.indexOf(id);
    if (a >= 0 && b >= 0) for (const item of list.slice(Math.min(a, b), Math.max(a, b) + 1)) state.selected.add(item);
  } else if (state.selected.has(id)) state.selected.delete(id); else state.selected.add(id);
  state.lastClicked = id; syncCards(); updateBulk();
}
function syncCards() { gallery.querySelectorAll('.card').forEach((card) => { const selected = state.selected.has(card.dataset.id); card.classList.toggle('selected', selected); card.querySelector('.select-check').textContent = selected ? '✓' : ''; }); }
function updateBulk() { $('bulkbar').hidden = state.selected.size === 0; $('selectedCount').textContent = state.selected.size; }
function recordUndo(changes) {
  const previous = {}; for (const id of Object.keys(changes)) previous[id] = state.labels[id] ?? null;
  state.undo.push(previous); if (state.undo.length > 100) state.undo.shift(); $('undoBtn').disabled = false;
}
async function assign(ids, category) {
  if (!ids.length) return;
  const before = assignedImages().filter((image) => state.labels[image.id]).length;
  const changes = Object.fromEntries(ids.map((id) => [id, category])); recordUndo(changes);
  for (const [id, value] of Object.entries(changes)) state.labels[id] = value;
  const after = assignedImages().filter((image) => state.labels[image.id]).length;
  state.selected.clear(); updateBulk(); updateStats(); renderGallery(); await save(changes);
  for (let milestone = Math.floor(before / 50) + 1; milestone <= Math.floor(after / 50); milestone++) celebrate(milestone * 50);
}
async function clearLabels(ids) {
  if (!ids.length) return; const changes = Object.fromEntries(ids.map((id) => [id, null])); recordUndo(changes);
  for (const id of ids) delete state.labels[id]; updateStats(); renderGallery(); await save(changes);
}
async function undo() {
  const previous = state.undo.pop(); if (!previous) return;
  for (const [id, category] of Object.entries(previous)) { if (category) state.labels[id] = category; else delete state.labels[id]; }
  $('undoBtn').disabled = state.undo.length === 0; updateStats(); renderGallery(); await save(previous);
}
async function save(changes) {
  const status = $('saveStatus'); status.textContent = '● Guardando…'; status.className = 'saving';
  try { const response = await fetch('/api/labels', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ changes }) }); if (!response.ok) throw new Error('save'); status.textContent = '● Guardado'; status.className = ''; }
  catch { status.textContent = '● Error al guardar'; status.className = 'error'; }
}
function openLightbox(image) { $('lightboxImage').src = `/full/${image.id}`; $('lightboxCaption').textContent = `${image.id} · ${image.path}`; $('lightbox').showModal(); }
function openQuick(filter = state.filter) {
  state.quickOrder = assignedImages().filter((image) => { const category = state.labels[image.id]; return filter === 'all' || (filter === 'pending' ? !category : category === filter); }).map((image) => image.id);
  if (!state.quickOrder.length && filter === 'pending') state.quickOrder = assignedImages().map((image) => image.id);
  const initial = state.quickOrder.findIndex((id) => !state.labels[id]); state.quickIndex = initial >= 0 ? initial : 0;
  if (!state.quickOrder.length) return; $('quickDialog').showModal(); showQuick();
}
function showQuick() {
  const id = state.quickOrder[state.quickIndex]; if (!id) return;
  $('quickImage').src = `/full/${id}`; $('quickImage').alt = `Imagen ${id}`; $('quickPosition').textContent = `${state.quickIndex + 1} / ${state.quickOrder.length} · ${id}`;
  const category = state.labels[id]; $('quickLabel').textContent = category ? `Actual: ${categories.find((c) => c.key === category).label}` : 'Pendiente';
  $('quickChoices').querySelectorAll('.choice').forEach((button) => {
    const selected = button.dataset.category === category;
    button.classList.toggle('chosen', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  $('quickPrev').disabled = state.quickIndex <= 0; $('quickNext').disabled = state.quickIndex >= state.quickOrder.length - 1;
}
async function classifyQuick(category) {
  if (state.classifying) return;
  const id = state.quickOrder[state.quickIndex]; if (!id) return;
  state.classifying = true;
  const label = categories.find((item) => item.key === category).label;
  $('quickFeedback').textContent = `✓ ${label} asignada`;
  $('quickFeedback').classList.remove('show'); void $('quickFeedback').offsetWidth; $('quickFeedback').classList.add('show');
  $('quickChoices').querySelectorAll('.choice').forEach((button) => button.classList.toggle('chosen', button.dataset.category === category));
  await assign([id], category);
  const nextPending = state.quickOrder.findIndex((candidate, index) => index > state.quickIndex && !state.labels[candidate]);
  if (nextPending >= 0) state.quickIndex = nextPending;
  else {
    const earlierPending = state.quickOrder.findIndex((candidate) => !state.labels[candidate]);
    if (earlierPending >= 0) state.quickIndex = earlierPending;
    else if (state.quickIndex < state.quickOrder.length - 1) state.quickIndex++;
  }
  window.setTimeout(() => { $('quickChoices').querySelectorAll('.choice').forEach((button) => button.classList.remove('chosen')); showQuick(); state.classifying = false; }, 260);
}
function quickMove(direction) { if (state.classifying) return; state.quickIndex = Math.max(0, Math.min(state.quickOrder.length - 1, state.quickIndex + direction)); showQuick(); }
function celebrate(milestone) {
  const inQuick = $('quickDialog').open;
  const total = assignedImages().length;
  const toast = $(inQuick ? 'quickMilestoneToast' : 'milestoneToast'); toast.textContent = `W ${state.rater} 🔥, llevas ${milestone} de #${total} imágenes❗`; toast.classList.add('visible');
  window.setTimeout(() => toast.classList.remove('visible'), 3000);
  const colors = ['#a3e635', '#75c9ee', '#ce9cff', '#ffc16d', '#f28d80', '#ffffff'];
  const layer = $(inQuick ? 'quickConfetti' : 'confetti'); layer.innerHTML = '';
  for (let i = 0; i < 110; i++) {
    const piece = document.createElement('i'); piece.style.left = `${Math.random() * 100}%`; piece.style.background = colors[i % colors.length];
    piece.style.setProperty('--fall', `${60 + Math.random() * 65}vh`); piece.style.setProperty('--spin', `${Math.random() * 1000 - 500}deg`); piece.style.animationDelay = `${Math.random() * 0.65}s`; layer.append(piece);
  }
  window.setTimeout(() => { layer.innerHTML = ''; }, 2600);
}

$('search').addEventListener('input', (event) => { state.query = event.target.value.trim(); renderGallery(); });
$('selectVisible').addEventListener('click', () => { currentImages().forEach((image) => state.selected.add(image.id)); syncCards(); updateBulk(); });
$('selectPending').addEventListener('click', () => { assignedImages().filter((image) => !state.labels[image.id]).forEach((image) => state.selected.add(image.id)); syncCards(); updateBulk(); });
$('clearSelection').addEventListener('click', () => { state.selected.clear(); syncCards(); updateBulk(); });
$('undoBtn').addEventListener('click', undo); $('quickStart').addEventListener('click', () => openQuick());
$('switchPerson').addEventListener('click', () => { if ($('quickDialog').open) $('quickDialog').close(); buildPeopleChoices(); $('welcomeDialog').showModal(); });
$('welcomeDialog').addEventListener('cancel', (event) => event.preventDefault());
$('quickClose').addEventListener('click', () => $('quickDialog').close()); $('quickPrev').addEventListener('click', () => quickMove(-1)); $('quickNext').addEventListener('click', () => quickMove(1));
$('lightboxClose').addEventListener('click', () => $('lightbox').close()); $('lightbox').addEventListener('click', (event) => { if (event.target === $('lightbox')) $('lightbox').close(); });
$('exportBtn').addEventListener('click', () => { const blob = new Blob([JSON.stringify({ dataset: 'Francesco/parasites-1s07h', variant: 'deduplicated', split: 'train', categories: categories.map((c) => c.key), items: state.images.map((image) => ({ id: image.id, path: image.path, category: state.labels[image.id] ?? null })) }, null, 2)], { type: 'application/json' }); const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = 'parasites_train_dedup_visual_labels.json'; link.click(); URL.revokeObjectURL(link.href); });
document.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') { event.preventDefault(); undo(); return; }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'a' && !$('quickDialog').open) { event.preventDefault(); currentImages().forEach((image) => state.selected.add(image.id)); syncCards(); updateBulk(); return; }
  if ($('quickDialog').open) {
    if (event.key >= '1' && event.key <= '5') { classifyQuick(categories[Number(event.key) - 1].key); event.preventDefault(); }
    else if (event.key === 'ArrowLeft') quickMove(-1); else if (event.key === 'ArrowRight') quickMove(1);
    else if (event.key === 'Escape') $('quickDialog').close();
    else if (event.key === 'Backspace') { clearLabels([state.quickOrder[state.quickIndex]]).then(() => showQuick()); event.preventDefault(); }
  } else if (!['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName) && event.key >= '1' && event.key <= '5' && state.selected.size) assign([...state.selected], categories[Number(event.key) - 1].key);
});
init().catch((error) => { console.error(error); $('progressText').textContent = 'Error al cargar el dataset'; });
