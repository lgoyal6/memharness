// Your volumes, the measured per-unit costs.
//
// The page's headline crossover is for the one conversation that was measured.
// The arithmetic generalises cleanly, because extraction is charged per turn
// and retrieval per question, so it is worth letting a reader put their own
// shape in rather than asking them to trust a number from somebody else's
// dataset. Every rate below is read out of runs.json, not typed in here.
(() => {
  const el = (id) => document.getElementById(id);
  let rates = null;

  const fmt = (n) =>
    n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : String(Math.round(n));

  function run() {
    if (!rates) return;
    const turns = Math.max(Number(el('c-turns').value) || 0, 1);
    const questions = Math.max(Number(el('c-questions').value) || 0, 1);
    const convos = Math.max(Number(el('c-convos').value) || 0, 1);

    const ingest = rates.ingestPerTurn * turns;
    const perQDelta = rates.kwPerQuestion - rates.memPerQuestion;
    // Where the one-off extraction is repaid by the cheaper retrieval.
    const cross = perQDelta > 0 ? ingest / perQDelta : Infinity;

    const memMonth = (ingest + rates.memPerQuestion * questions) * convos;
    const kwMonth = rates.kwPerQuestion * questions * convos;

    el('c-cross').textContent = Number.isFinite(cross) ? `${Math.round(cross)} questions` : 'never';
    el('c-asked').textContent = `${questions} questions`;
    el('c-mem').textContent = `${fmt(memMonth)} tokens`;
    el('c-kw').textContent = `${fmt(kwMonth)} tokens`;

    const memWins = memMonth < kwMonth;
    el('c-mem').className = memWins ? 'good' : 'bad';
    el('c-kw').className = memWins ? '' : 'good';

    const b = el('calc-banner');
    if (memWins) {
      b.className = 'banner calm';
      b.textContent =
        `Worth it. You ask ${questions} against a crossover of ${Math.round(cross)}, so memory ` +
        `costs ${fmt(kwMonth - memMonth)} fewer tokens a month than keyword search.`;
    } else {
      const ratio = memMonth / Math.max(kwMonth, 1);
      b.className = 'banner alarm';
      b.textContent =
        `Not worth it on cost. You ask ${questions} and the crossover is ${Math.round(cross)}, ` +
        `so you would pay ${ratio.toFixed(1)}x what keyword search costs. It may still be worth ` +
        `it on accuracy, which is the other half of this page.`;
    }
  }

  fetch('./data/runs.json')
    .then((r) => r.json())
    .then((d) => {
      const mem = d.systems.find((s) => s.key === 'mem0');
      const kw = d.systems.find((s) => s.key === 'bm25');
      rates = {
        ingestPerTurn: mem.ingest_tokens / d.dataset.turns,
        memPerQuestion: mem.per_q_tokens,
        kwPerQuestion: kw.per_q_tokens,
      };
      el('calc-src').textContent =
        `${rates.ingestPerTurn.toFixed(1)} tokens a turn to read, ` +
        `${Math.round(rates.memPerQuestion)} against ${Math.round(rates.kwPerQuestion)} a question`;
      ['c-turns', 'c-questions', 'c-convos'].forEach((id) =>
        el(id).addEventListener('input', run));
      run();
    })
    .catch(() => {
      el('calc-banner').className = 'banner alarm';
      el('calc-banner').textContent = 'Could not load the measured rates.';
    });
})();
