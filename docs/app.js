// Draws docs/data/runs.json, which scripts/make_page_data.py takes out of the
// two run files in results/. The only arithmetic done here is the amortisation
// itself: cost at Q questions is the one-off extraction plus Q times the
// measured per-question cost. Nothing is fitted.

const el = (id) => document.getElementById(id);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const state = { data: null, unit: 'tokens', q: 20, scope: 'pair' };

// Full context costs an order of magnitude more per question, so plotting all
// four flattens the two that are actually being compared into one line at the
// bottom. The comparison that matters is the default; the rest is a click away.
const SCOPES = {
  pair: { label: 'mem0 against keyword search', keep: (s) => s.key === 'mem0' || s.key === 'bm25' },
  all: { label: 'all four', keep: () => true },
};
const shown = () => state.data.systems.filter(SCOPES[state.scope].keep);

const UNITS = {
  tokens: {
    label: 'Tokens',
    once: (s) => s.ingest_tokens,
    each: (s) => s.per_q_tokens,
    fmt: (v) => (v >= 1e6 ? `${(v / 1e6).toFixed(2)}M` : Math.round(v).toLocaleString('en-US')),
  },
  usd: {
    label: 'Dollars',
    once: (s) => s.ingest_usd,
    each: (s) => s.per_q_usd,
    fmt: (v) => `$${v < 0.01 ? v.toFixed(5) : v.toFixed(3)}`,
  },
};

const unit = () => UNITS[state.unit];
const costAt = (s, q) => unit().once(s) + unit().each(s) * q;

// Where two lines cross, or null when the cheaper one starts cheaper and the
// gradients never let it be caught.
function crossover(a, b) {
  const dOnce = unit().once(a) - unit().once(b);
  const dEach = unit().each(b) - unit().each(a);
  if (dEach <= 0 || dOnce <= 0) return null;
  return dOnce / dEach;
}

function labelOnPaper(ctx, text, x, y, align = 'center') {
  const w = ctx.measureText(text).width;
  const left = align === 'center' ? x - w / 2 : align === 'right' ? x - w : x;
  const prev = ctx.fillStyle;
  ctx.fillStyle = css('--paper');
  ctx.fillRect(left - 3, y - 11, w + 6, 14);
  ctx.fillStyle = prev;
  ctx.textAlign = align;
  ctx.fillText(text, x, y);
}

function fitCanvas(canvas, h0) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w0 = canvas.clientWidth || 1200;
  canvas.width = Math.round(w0 * dpr);
  canvas.height = Math.round(h0 * dpr);
  canvas.style.height = h0 + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w0, h0);
  return { ctx, w: w0, h: h0 };
}

function drawAmort() {
  const sys = shown();
  const { ctx, w, h } = fitCanvas(el('plot-amort'), 300);
  const pad = { l: 74, r: 176, t: 20, b: 48 };
  const iw = w - pad.l - pad.r;
  const ih = h - pad.t - pad.b;
  const maxQ = Number(el('qs').max);
  const top = Math.max(...sys.map((s) => costAt(s, maxQ))) * 1.06;
  const X = (q) => pad.l + (q / maxQ) * iw;
  const Y = (v) => pad.t + ih - (v / top) * ih;

  ctx.strokeStyle = css('--hair');
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + ih); ctx.lineTo(pad.l + iw, pad.t + ih);
  ctx.stroke();
  ctx.font = "11px 'Courier New', monospace";
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = (top / 4) * i;
    ctx.fillStyle = css('--faint');
    ctx.fillText(unit().fmt(v), pad.l - 8, Y(v) + 3);
    if (i) {
      ctx.strokeStyle = '#e8e3d6';
      ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(pad.l + iw, Y(v)); ctx.stroke();
    }
  }
  ctx.textAlign = 'center';
  for (let q = 0; q <= maxQ; q += maxQ / 6) {
    ctx.fillStyle = css('--faint');
    ctx.fillText(String(Math.round(q)), X(q), pad.t + ih + 16);
  }
  ctx.fillStyle = css('--faint');
  ctx.fillText('questions asked against this one conversation', pad.l + iw / 2, h - 8);

  // Dash pattern rather than colour, so four lines stay separable in print.
  const DASH = [[], [7, 4], [2, 3], [10, 3, 2, 3]];
  sys.forEach((s, i) => {
    ctx.save();
    ctx.setLineDash(DASH[i % DASH.length]);
    ctx.strokeStyle = css('--ox');
    ctx.lineWidth = i === 0 ? 2.4 : 1.5;
    ctx.beginPath();
    ctx.moveTo(X(0), Y(costAt(s, 0)));
    ctx.lineTo(X(maxQ), Y(costAt(s, maxQ)));
    ctx.stroke();
    ctx.restore();
    ctx.textAlign = 'left';
    ctx.font = "12px 'Times New Roman', serif";
    ctx.fillStyle = css('--sub');
    ctx.fillText(s.label, pad.l + iw + 10, Y(costAt(s, maxQ)) + 4);
  });

  const cross = crossover(state.data.systems[0], state.data.systems[1]);
  if (cross && cross <= maxQ) {
    ctx.save();
    ctx.strokeStyle = css('--bad');
    ctx.lineWidth = 1.4;
    ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(X(cross), pad.t); ctx.lineTo(X(cross), pad.t + ih); ctx.stroke();
    ctx.restore();
    ctx.font = "12px 'Times New Roman', serif";
    ctx.fillStyle = css('--bad');
    labelOnPaper(ctx, `mem0 overtakes here: ${Math.round(cross)}`, X(cross), pad.t + 12);
  }

  // Where the reader currently is.
  ctx.save();
  ctx.strokeStyle = css('--ink');
  ctx.setLineDash([2, 3]);
  ctx.beginPath(); ctx.moveTo(X(state.q), pad.t); ctx.lineTo(X(state.q), pad.t + ih); ctx.stroke();
  ctx.restore();
  sys.forEach((s) => {
    ctx.beginPath();
    ctx.arc(X(state.q), Y(costAt(s, state.q)), 3.5, 0, Math.PI * 2);
    ctx.fillStyle = css('--ox');
    ctx.fill();
  });
}

function renderAmort() {
  const [mem0, bm25] = state.data.systems;
  const q = state.q;
  el('r-q').textContent = q.toLocaleString('en-US');
  el('r-mem0').textContent = unit().fmt(costAt(mem0, q));
  el('r-bm25').textContent = unit().fmt(costAt(bm25, q));
  const cheaper = costAt(mem0, q) <= costAt(bm25, q) ? mem0 : bm25;
  el('r-winner').textContent = cheaper.label;
  el('cap-what').textContent =
    `${state.data.dataset.turns} turns, ${state.data.dataset.questions_used} questions measured`;
  el('cap-price').textContent =
    state.unit === 'usd' ? `${state.data.pricing.sheet_name}, a config input` : 'measured token counts';
  drawAmort();

  const cross = crossover(mem0, bm25);
  const b = el('amort-banner');
  const ratio = costAt(cheaper === mem0 ? bm25 : mem0, q) / costAt(cheaper, q);
  if (cheaper === mem0) {
    b.className = 'banner calm';
    b.textContent =
      `At ${q} questions mem0 has paid off its extraction and is ${ratio.toFixed(2)}x cheaper ` +
      `than keyword search.`;
  } else {
    b.className = 'banner alarm';
    b.textContent =
      `At ${q} questions mem0 costs ${ratio.toFixed(1)}x what keyword search does, and stays ` +
      `dearer until about ${Math.round(cross)}.`;
  }
}

function board() {
  const head =
    '<tr><th>system</th><th>accuracy</th><th>strict</th><th>read once</th>' +
    '<th>per question</th><th>retrieve p50</th><th>context sent</th></tr>';
  const body = state.data.systems
    .map(
      (s) =>
        `<tr><td class="sys">${s.label}</td>` +
        `<td class="num ${s.accuracy >= 0.6 ? 'good' : s.accuracy < 0.2 ? 'bad' : ''}">${s.accuracy.toFixed(2)}</td>` +
        `<td class="num">${s.accuracy_strict.toFixed(2)}</td>` +
        `<td class="num ${s.ingest_tokens ? 'bad' : ''}">${s.ingest_tokens ? s.ingest_tokens.toLocaleString('en-US') : 'nothing'}</td>` +
        `<td class="num">${Math.round(s.per_q_tokens).toLocaleString('en-US')}</td>` +
        `<td class="num ${s.p50_retrieve_ms > 10 ? 'bad' : ''}">${s.p50_retrieve_ms.toFixed(2)} ms</td>` +
        `<td class="num">${Math.round(s.mean_context_chars).toLocaleString('en-US')}</td></tr>`,
    )
    .join('');
  el('board').innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
  el('board-banner').textContent =
    'Accuracy is the LLM judge; strict is exact-match. Tokens are measured, and "read once" is ' +
    'charged before a single question is asked.';
}

function picker(node, items, current, onPick) {
  node.innerHTML = '';
  items.forEach(({ key, label }) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.setAttribute('aria-pressed', String(key === current()));
    b.addEventListener('click', () => {
      onPick(key);
      [...node.children].forEach((c) => c.setAttribute('aria-pressed', String(c === b)));
    });
    node.appendChild(b);
  });
}

async function main() {
  const res = await fetch('./data/runs.json');
  if (!res.ok) {
    el('amort-banner').textContent = `Could not load the runs (HTTP ${res.status}).`;
    return;
  }
  state.data = await res.json();

  picker(
    el('scope'),
    Object.entries(SCOPES).map(([k, v]) => ({ key: k, label: v.label })),
    () => state.scope,
    (k) => { state.scope = k; renderAmort(); },
  );
  picker(
    el('unit'),
    Object.entries(UNITS).map(([k, v]) => ({ key: k, label: v.label })),
    () => state.unit,
    (k) => { state.unit = k; renderAmort(); },
  );
  const qs = el('qs');
  qs.addEventListener('input', (e) => { state.q = Number(e.target.value); renderAmort(); });
  window.addEventListener('resize', renderAmort);

  renderAmort();
  board();
}

main();
