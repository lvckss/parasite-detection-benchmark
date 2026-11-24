const transformNames = {
  horizontal_flip: 'Flip horizontal',
  vertical_flip: 'Flip vertical',
  rotation: 'Rotación',
  scale: 'Escala',
  shear: 'Cizallamiento',
  crop: 'Recorte',
  perspective: 'Perspectiva',
  elastic: 'Deformación elástica',
  brightness: 'Brillo',
  contrast: 'Contraste',
  hue_saturation: 'Tono / saturación',
  channel_shuffle: 'Mezcla de canales',
  gaussian_noise: 'Ruido gaussiano',
  gaussian_blur: 'Desenfoque gaussiano',
  average_blur: 'Desenfoque promedio',
  sharpen: 'Enfoque',
  dropout: 'Dropout',
};

const state = { images: [], labels: {}, filtered: [], rendered: 0 };
const $ = (id) => document.getElementById(id);
const gallery = $('gallery');
const sentinel = $('sentinel');
let galleryObserver;

async function init() {
  await refreshLabels();
}

async function refreshLabels() {
  const button = $('refreshLabels');
  button.disabled = true;
  try {
    const response = await fetch('/api/images');
    if (!response.ok) throw new Error(`No se pudo cargar el índice (${response.status}).`);
    const data = await response.json();
    state.images = data.images;
    state.labels = data.labels;
    $('imageCount').textContent = state.images.length.toLocaleString('es');
    $('classifiedCount').textContent = state.images.filter((image) => image.label).length.toLocaleString('es');
    $('error').hidden = true;
    buildCategoryFilter();
    renderGallery();
  } finally {
    button.disabled = false;
  }
}

function buildCategoryFilter() {
  const counts = Object.fromEntries(Object.keys(state.labels).map((key) => [key, 0]));
  for (const image of state.images) if (image.label) counts[image.label]++;
  $('categoryFilter').innerHTML = '<option value="all">Todas</option>' +
    Object.entries(state.labels).map(([key, name]) =>
      `<option value="${key}">${name} · ${counts[key]}</option>`).join('') +
    `<option value="unclassified">Sin clasificar · ${state.images.length - state.images.filter((image) => image.label).length}</option>`;
}

function filteredImages() {
  const query = $('search').value.trim().toLowerCase();
  const category = $('categoryFilter').value;
  return state.images.filter((image) => {
    const matchesText = !query || image.stem.toLowerCase().includes(query);
    const matchesCategory = category === 'all' ||
      (category === 'unclassified' ? !image.label : image.label === category);
    return matchesText && matchesCategory;
  });
}

function renderGallery() {
  if (galleryObserver) galleryObserver.disconnect();
  gallery.replaceChildren();
  state.filtered = filteredImages();
  state.rendered = 0;
  $('visibleCount').textContent = `${state.filtered.length.toLocaleString('es')} resultados`;
  $('emptyState').hidden = state.filtered.length !== 0;
  appendNext();
  galleryObserver = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) appendNext();
  }, { rootMargin: '400px' });
  galleryObserver.observe(sentinel);
}

function appendNext() {
  const fragment = document.createDocumentFragment();
  const end = Math.min(state.rendered + 72, state.filtered.length);
  while (state.rendered < end) fragment.append(makeCard(state.filtered[state.rendered++]));
  gallery.append(fragment);
  sentinel.hidden = state.rendered >= state.filtered.length;
}

function makeCard(record) {
  const card = document.createElement('button');
  card.className = 'image-card';
  card.type = 'button';
  card.addEventListener('click', () => openComparison(record));

  const image = document.createElement('img');
  image.src = `/image/${encodeURIComponent(record.id)}`;
  image.alt = `Original ${record.stem}`;
  image.loading = 'lazy';

  const info = document.createElement('span');
  info.className = 'card-info';
  const id = document.createElement('b');
  id.textContent = record.stem;
  const label = document.createElement('span');
  label.className = record.label ? `badge ${record.label}` : 'badge unclassified';
  label.textContent = record.label ? state.labels[record.label] : 'Sin clasificar';
  const variants = document.createElement('small');
  variants.textContent = `${record.variants.length} variantes`;
  info.append(id, label, variants);
  card.append(image, info);
  return card;
}

function openComparison(record) {
  $('dialogTitle').textContent = record.stem;
  const classification = $('classification');
  classification.className = record.label ? `classification ${record.label}` : 'classification unclassified';
  classification.textContent = record.label
    ? `Clasificación visual: ${state.labels[record.label]}`
    : 'Sin clasificación visual';

  const grid = $('comparisonGrid');
  grid.replaceChildren();
  grid.append(makeComparisonPanel('Original', record.id, []));
  for (const variant of record.variants) {
    grid.append(makeComparisonPanel(`Variante ${variant.index}`, variant.id, variant.transforms));
  }
  $('compareDialog').showModal();
}

function makeComparisonPanel(title, imageId, transforms) {
  const panel = document.createElement('article');
  panel.className = 'comparison-panel';
  const heading = document.createElement('h3');
  heading.textContent = title;
  const image = document.createElement('img');
  image.src = `/image/${encodeURIComponent(imageId)}`;
  image.alt = `${title}: ${imageId}`;
  image.loading = 'lazy';
  const caption = document.createElement('small');
  caption.textContent = imageId;
  const badges = document.createElement('div');
  badges.className = 'transform-list';
  if (transforms.length) {
    for (const transform of transforms) {
      const badge = document.createElement('span');
      badge.className = 'transform-badge';
      badge.textContent = transformNames[transform] || transform;
      badges.append(badge);
    }
  } else {
    const none = document.createElement('span');
    none.className = 'no-transforms';
    none.textContent = 'Sin transformaciones';
    badges.append(none);
  }
  panel.append(heading, image, caption, badges);
  return panel;
}

$('search').addEventListener('input', renderGallery);
$('categoryFilter').addEventListener('change', renderGallery);
$('refreshLabels').addEventListener('click', () => {
  refreshLabels().catch((error) => {
    console.error(error);
    $('error').textContent = error.message;
    $('error').hidden = false;
  });
});
$('dialogClose').addEventListener('click', () => $('compareDialog').close());
$('compareDialog').addEventListener('click', (event) => {
  if (event.target === $('compareDialog')) $('compareDialog').close();
});

init().catch((error) => {
  console.error(error);
  $('error').textContent = error.message;
  $('error').hidden = false;
});
