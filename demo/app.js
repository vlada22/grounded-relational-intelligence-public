const data = await fetch('./data/results.json', { cache: 'no-store' }).then(response => {
  if (!response.ok) throw new Error(`results: ${response.status}`);
  return response.json();
});

const $ = selector => document.querySelector(selector);
const model = $('#model');
const layer = $('#layer');
const grid = $('#grid');
const gridLabel = $('#grid-label');
const canvas = $('#overlay');
const scene = $('#scene');
const metrics = {
  retrieval: 'Retrieval',
  ari: 'Cluster ARI',
  boundary_f1: 'Boundary F1',
  grouping_stability: 'Grouping stability',
};
const modelAccents = {
  DINOv2: '#70e0bd',
  DINOv3: '#8fb5ff',
  'SigLIP 2': '#f1c875',
};

let metric = 'retrieval';
let row = 16;
let col = 16;

model.innerHTML = Object.keys(data.models).map(value => `<option>${value}</option>`).join('');
layer.innerHTML = data.protocol.layers.map(value => `<option value="${value}">Block ${value}</option>`).join('');
$('#metric-switch').innerHTML = Object.entries(metrics)
  .map(([key, label], index) => `<button type="button" data-metric="${key}" class="${index === 0 ? 'active' : ''}" aria-pressed="${index === 0}">${label}</button>`)
  .join('');

model.addEventListener('change', () => {
  preserveSourceCoordinate();
  render();
});
layer.addEventListener('change', render);
grid.addEventListener('change', drawPatch);
$('#metric-switch').addEventListener('click', event => {
  const button = event.target.closest('button');
  if (!button) return;
  metric = button.dataset.metric;
  document.querySelectorAll('#metric-switch button').forEach(item => {
    const active = item === button;
    item.classList.toggle('active', active);
    item.setAttribute('aria-pressed', String(active));
  });
  renderChart();
});
scene.addEventListener('click', event => {
  const bounds = scene.getBoundingClientRect();
  const [rows, cols] = current().grid;
  col = clamp(Math.floor((event.clientX - bounds.left) / bounds.width * cols), 0, cols - 1);
  row = clamp(Math.floor((event.clientY - bounds.top) / bounds.height * rows), 0, rows - 1);
  drawPatch();
});
scene.addEventListener('keydown', event => {
  const delta = {
    ArrowLeft: [0, -1],
    ArrowRight: [0, 1],
    ArrowUp: [-1, 0],
    ArrowDown: [1, 0],
  }[event.key];
  if (!delta) return;
  event.preventDefault();
  const [rows, cols] = current().grid;
  row = clamp(row + delta[0], 0, rows - 1);
  col = clamp(col + delta[1], 0, cols - 1);
  drawPatch();
});

function current() {
  return data.models[model.value];
}

function preserveSourceCoordinate() {
  const previous = scene.dataset.grid ? scene.dataset.grid.split('x').map(Number) : [32, 32];
  const x = (col + 0.5) / previous[1];
  const y = (row + 0.5) / previous[0];
  const [rows, cols] = current().grid;
  col = clamp(Math.floor(x * cols), 0, cols - 1);
  row = clamp(Math.floor(y * rows), 0, rows - 1);
}

function setAccent() {
  const accent = modelAccents[model.value] || '#70e0bd';
  document.documentElement.style.setProperty('--accent', accent);
  $('#model-chip').textContent = model.value;
}

function render() {
  setAccent();
  const record = current().layers[layer.value];
  $('#retrieval').textContent = record.retrieval.toFixed(3);
  $('#ari').textContent = record.ari.toFixed(3);
  $('#boundary').textContent = record.boundary_f1.toFixed(3);
  $('#stability').textContent = record.grouping_stability.toFixed(3);
  const [rows, cols] = current().grid;
  scene.dataset.grid = `${rows}x${cols}`;
  gridLabel.textContent = `${rows}×${cols} patch grid`;
  renderChart();
  renderRelations();
  drawPatch();
}

function renderChart() {
  const svg = $('#chart');
  const currentModel = current();
  const layers = data.protocol.layers;
  const values = layers.map(value => currentModel.layers[String(value)][metric]);
  const accent = modelAccents[model.value] || '#70e0bd';
  const x0 = 54;
  const x1 = 590;
  const y0 = 235;
  const y1 = 35;
  let markup = '';

  for (const tick of [0, .25, .5, .75, 1]) {
    const y = y0 - (y0 - y1) * tick;
    markup += `<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="rgba(220,240,235,.10)"/>`;
    markup += `<text x="8" y="${y + 4}" fill="#71868a" font-size="11">${tick.toFixed(2)}</text>`;
  }

  const points = values.map((value, index) => [
    x0 + (x1 - x0) * index / 2,
    y0 - value * (y0 - y1),
    value,
  ]);

  markup += `<polyline points="${points.map(point => `${point[0]},${point[1]}`).join(' ')}" fill="none" stroke="${accent}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>`;
  points.forEach((point, index) => {
    markup += `<circle cx="${point[0]}" cy="${point[1]}" r="6" fill="#081115" stroke="${accent}" stroke-width="4"/>`;
    markup += `<text x="${point[0] + 10}" y="${point[1] - 10}" fill="#edf6f3" font-size="11">${point[2].toFixed(3)}</text>`;
    markup += `<text x="${point[0] - 7}" y="260" fill="#98aaad" font-size="12">L${layers[index]}</text>`;
  });

  svg.innerHTML = markup;
  svg.setAttribute('aria-label', `${metrics[metric]} across transformer depth for ${model.value}`);
}

function renderRelations() {
  const record = current().typed_relationship_selected;
  $('#relations').innerHTML = [
    ['Best observed base layer', `L${record.layer}`],
    ['Matched truth-node recall', record.node_recall.toFixed(3)],
    ['Adjacent F1', record.adjacent_f1.toFixed(3)],
    ['Near F1', record.near_f1.toFixed(3)],
    ['Embedding diagnostic F1', record.embedding_similarity_diagnostic_f1.toFixed(3)],
    ['Diagnostic macro F1', record.macro_f1.toFixed(3)],
  ].map(([label, value]) => `<div class="relation"><span>${label}</span><b>${value}</b></div>`).join('');
  $('#relation-note').textContent = 'The displayed layer is a post-hoc descriptive best-observed base-scene summary. Embedding similarity is a cosine diagnostic, not a semantic ground-truth edge type.';
}

function drawPatch() {
  const context = canvas.getContext('2d');
  const [rows, cols] = current().grid;
  const patchWidth = 448 / cols;
  const patchHeight = 448 / rows;
  context.clearRect(0, 0, 448, 448);
  if (grid.checked) {
    context.beginPath();
    for (let index = 1; index < cols; index += 1) {
      const position = index * patchWidth;
      context.moveTo(position, 0);
      context.lineTo(position, 448);
    }
    for (let index = 1; index < rows; index += 1) {
      const position = index * patchHeight;
      context.moveTo(0, position);
      context.lineTo(448, position);
    }
    context.strokeStyle = 'rgba(232,247,242,.22)';
    context.lineWidth = 1;
    context.stroke();
  }
  const accent = modelAccents[model.value] || '#70e0bd';
  context.strokeStyle = '#ffffff';
  context.lineWidth = 4;
  context.strokeRect(col * patchWidth + 2, row * patchHeight + 2, Math.max(2, patchWidth - 4), Math.max(2, patchHeight - 4));
  context.strokeStyle = accent;
  context.lineWidth = 2;
  context.strokeRect(col * patchWidth + 5, row * patchHeight + 5, Math.max(2, patchWidth - 10), Math.max(2, patchHeight - 10));
  const x0 = Math.round(col * patchWidth);
  const y0 = Math.round(row * patchHeight);
  const x1 = Math.round((col + 1) * patchWidth);
  const y1 = Math.round((row + 1) * patchHeight);
  $('#patch').textContent = `row ${row} · col ${col}`;
  $('#box').textContent = `[${x0}, ${y0}] → [${x1}, ${y1}]`;
  scene.setAttribute('aria-label', `Controlled scene. ${model.value} ${rows} by ${cols} grid. Selected patch row ${row}, column ${col}. Arrow keys move selection.`);
}

function clamp(value, minimum, maximum) {
  return Math.min(Math.max(value, minimum), maximum);
}

render();
