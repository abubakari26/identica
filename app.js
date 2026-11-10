/* Identica: client-side fish identification that learns from your corrections.
 *
 * How it works
 * 1. MobileNet (pre-trained, TensorFlow.js) turns each photo into an "embedding":
 *    a list of numbers describing what the image looks like.
 * 2. Every fish you confirm or correct is saved with its embedding (IndexedDB).
 * 3. A new photo is compared with the saved embeddings (cosine similarity, k-nearest
 *    neighbours). The closest species win the vote.
 *
 * MobileNet was NOT trained on Ghanaian fish, so Identica starts with no species
 * knowledge. It becomes useful as you add labelled photos of each species.
 */
(() => {
  'use strict';

  /* ---------- Settings ---------- */
  const CFG = {
    K: 5,                    // neighbours that vote
    STRONG_MATCH: 0.85,      // best similarity at or above this = strong match
    MIN_MATCH: 0.72,         // below this = unfamiliar image
    MODEL: { version: 2, alpha: 1.0 },
    DB_URL: 'database.json', // reference species data (optional at this stage)
    IDB_NAME: 'identica',
    IDB_STORE: 'examples',
    SW_URL: 'sw.js'          // service worker (added in the next step)
  };

  const OPTIONS = {
    body: ['Fusiform', 'Laterally compressed', 'Depressed (flattened top to bottom)', 'Eel-like', 'Elongate', 'Globiform'],
    mouth: ['Terminal', 'Sub-terminal', 'Superior', 'Inferior'],
    tail: ['Forked', 'Rounded', 'Crescent-shaped (lunate)', 'Truncated', 'Emarginate', 'Pointed'],
    rakers: [
      ['not-visible', 'Not visible'],
      ['short-thick', 'Few, short and thick'],
      ['intermediate', 'Intermediate'],
      ['long-fine', 'Many, long and fine']
    ],
    sex: ['Unknown', 'Male', 'Female'],
    age: ['Unknown', 'Juvenile', 'Sub-adult', 'Adult']
  };

  const FEEDING_HINT = {
    'not-visible': '',
    'short-thick': 'Short, thick rakers usually go with predatory or omnivorous feeding.',
    'intermediate': 'Intermediate rakers usually go with omnivorous or detritus feeding.',
    'long-fine': 'Long, fine rakers usually go with filter feeding on plankton.'
  };

  /* ---------- State ---------- */
  const state = {
    model: null,
    modelPromise: null,
    species: [],      // reference species from database.json
    records: [],      // saved examples {id,label,embedding,traits,ts}
    embOrig: null,
    embFlip: null,
    result: null,
    saved: false
  };

  const $ = (id) => document.getElementById(id);

  /* ---------- Small UI helpers ---------- */
  function setStatus(msg, kind) {
    const el = $('status');
    el.textContent = msg || '';
    el.className = 'mt-3 min-h-[1.25rem] text-sm ' + (kind === 'error' ? 'font-medium text-laterite' : 'text-ink/80');
  }

  let toastTimer;
  function toast(msg) {
    const el = $('toast');
    el.textContent = msg;
    el.classList.remove('hidden');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add('hidden'), 2800);
  }

  function setBusy(on) {
    $('busy').classList.toggle('hidden', !on);
    $('btnCamera').disabled = on;
    $('btnGallery').disabled = on;
  }

  const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

  function fillSelect(sel, items) {
    sel.textContent = '';
    items.forEach((it) => {
      const o = document.createElement('option');
      if (Array.isArray(it)) { o.value = it[0]; o.textContent = it[1]; }
      else { o.value = it; o.textContent = it; }
      sel.appendChild(o);
    });
  }

  /* ---------- IndexedDB (saved examples) ---------- */
  function idb() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(CFG.IDB_NAME, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(CFG.IDB_STORE, { keyPath: 'id' });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  async function idbTx(mode, fn) {
    const db = await idb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(CFG.IDB_STORE, mode);
      const store = tx.objectStore(CFG.IDB_STORE);
      const out = fn(store);
      tx.oncomplete = () => resolve(out && out.result !== undefined ? out.result : undefined);
      tx.onerror = () => reject(tx.error);
    });
  }
  const idbAll = () => idbTx('readonly', (s) => s.getAll());
  const idbPut = (rec) => idbTx('readwrite', (s) => s.put(rec));
  const idbClear = () => idbTx('readwrite', (s) => s.clear());

  /* ---------- Reference species ---------- */
  const speciesLabel = (sp) => `${sp.common_name} (${sp.scientific_name})`;

  function findSpecies(label) {
    if (!label) return null;
    const l = label.toLowerCase();
    return state.species.find((sp) => speciesLabel(sp).toLowerCase() === l)
      || state.species.find((sp) => l.includes(sp.scientific_name.toLowerCase()))
      || null;
  }

  function splitLabel(label) {
    const m = /^(.*?)\s*\((.+)\)\s*$/.exec(label || '');
    return m ? { common: m[1], sci: m[2] } : { common: label || '', sci: '' };
  }

  async function loadReferenceData() {
    try {
      const res = await fetch(CFG.DB_URL, { cache: 'no-cache' });
      if (!res.ok) throw new Error(res.status);
      const data = await res.json();
      state.species = Array.isArray(data.species) ? data.species : [];
    } catch (_) {
      state.species = []; // The app still works: species names are typed in.
    }
    const list = $('speciesList');
    list.textContent = '';
    state.species.forEach((sp) => {
      const o = document.createElement('option');
      o.value = speciesLabel(sp);
      list.appendChild(o);
    });
  }

  /* ---------- Model ---------- */
  function ensureModel() {
    if (state.model) return Promise.resolve(state.model);
    if (!state.modelPromise) {
      setStatus('Loading the vision model. The first load downloads about 14 MB.');
      state.modelPromise = tf.ready()
        .then(() => mobilenet.load(CFG.MODEL))
        .then((m) => { state.model = m; setStatus(''); return m; })
        .catch((err) => {
          state.modelPromise = null;
          throw new Error('The vision model could not load. Check your internet connection and try again.');
        });
    }
    return state.modelPromise;
  }

  /* Letterbox into a 224x224 square so the fish keeps its true proportions. */
  function letterbox(src, flip) {
    const S = 224;
    const c = document.createElement('canvas');
    c.width = c.height = S;
    const x = c.getContext('2d');
    x.fillStyle = '#808080';
    x.fillRect(0, 0, S, S);
    const r = Math.min(S / src.width, S / src.height);
    const w = src.width * r, h = src.height * r;
    x.save();
    if (flip) { x.translate(S, 0); x.scale(-1, 1); }
    x.drawImage(src, (S - w) / 2, (S - h) / 2, w, h);
    x.restore();
    return c;
  }

  /* Image to a unit-length embedding (so a dot product equals cosine similarity). */
  function embed(canvas) {
    return tf.tidy(() => {
      const e = state.model.infer(canvas, true);
      return Array.from(tf.div(e, tf.norm(e)).dataSync());
    });
  }

  const dot = (a, b) => {
    let s = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) s += a[i] * b[i];
    return s;
  };

  /* ---------- Classification (k-nearest neighbours) ---------- */
  function classify(emb) {
    if (!state.records.length) return null;
    const scored = state.records
      .map((r) => ({ label: r.label, sim: dot(emb, r.embedding) }))
      .sort((a, b) => b.sim - a.sim);
    const top = scored.slice(0, CFG.K);

    const votes = new Map();
    let sum = 0;
    top.forEach((t) => {
      const w = Math.pow(Math.max(t.sim, 0), 8);
      votes.set(t.label, (votes.get(t.label) || 0) + w);
      sum += w;
    });
    const ranked = [...votes].map(([label, w]) => ({
      label,
      share: sum ? w / sum : 0,
      best: Math.max(...top.filter((t) => t.label === label).map((t) => t.sim))
    })).sort((a, b) => b.share - a.share);

    return { ranked, bestSim: scored[0].sim };
  }

  /* ---------- Photo handling ---------- */
  async function loadBitmap(file) {
    if ('createImageBitmap' in window) {
      try { return await createImageBitmap(file, { imageOrientation: 'from-image' }); } catch (_) { /* fall back */ }
    }
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('This file is not an image the browser can open.')); };
      img.src = url;
    });
  }

  function drawPreview(src) {
    const c = $('preview');
    const maxW = 900;
    const scale = Math.min(1, maxW / src.width);
    c.width = Math.round(src.width * scale);
    c.height = Math.round(src.height * scale);
    c.getContext('2d').drawImage(src, 0, 0, c.width, c.height);
    c.classList.remove('hidden');
    $('placeholder').classList.add('hidden');
    $('frame').classList.remove('border-dashed');
  }

  async function handleFile(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) { setStatus('Choose an image file (JPG, PNG or WebP).', 'error'); return; }

    setBusy(true);
    state.saved = false;
    try {
      setStatus('Reading photo…');
      const bmp = await loadBitmap(file);
      drawPreview(bmp);

      await ensureModel();
      setStatus('Analysing…');
      await nextFrame();

      state.embOrig = embed(letterbox(bmp, false));
      state.embFlip = embed(letterbox(bmp, true));   // mirrored copy, saved when you teach the app
      state.result = classify(state.embOrig);

      renderResult();
      setStatus('');
    } catch (err) {
      console.error(err);
      setStatus(err.message || 'Could not analyse this photo. Try another one.', 'error');
    } finally {
      setBusy(false);
    }
  }

  /* ---------- Rendering ---------- */
  function renderResult() {
    const r = state.result;
    $('resultCard').hidden = false;
    $('traitCard').hidden = false;
    $('teachCard').hidden = false;
    $('btnSave').disabled = false;
    $('btnSave').textContent = 'Save to training set';

    if (!r) {
      $('resCommon').textContent = 'No species learned yet';
      $('resSci').textContent = '';
      $('confWrap').hidden = true;
      $('altWrap').hidden = true;
      $('resNote').textContent = 'Identica learns from your confirmed fish. Enter the species below and save this photo as the first example.';
      $('speciesInput').value = '';
      applyTraits(null);
      return;
    }

    const top = r.ranked[0];
    const { common, sci } = splitLabel(top.label);
    $('resCommon').textContent = common;
    $('resSci').textContent = sci;
    $('confWrap').hidden = false;

    let tier, note;
    if (r.bestSim >= CFG.STRONG_MATCH) {
      tier = 'Strong match';
      note = 'This photo closely resembles fish you have already saved as this species.';
    } else if (r.bestSim >= CFG.MIN_MATCH) {
      tier = 'Possible match';
      note = 'Similar to your saved photos, but not close. Check the features below before you accept it.';
    } else {
      tier = 'Unfamiliar fish';
      note = 'This does not look like anything saved so far. Treat the name as a guess and teach Identica the right species.';
    }
    const pct = Math.round(top.share * Math.min(1, r.bestSim) * 100);
    $('confLabel').textContent = tier;
    $('confPct').textContent = pct + '%';
    $('confBar').style.width = pct + '%';
    $('confBar').className = 'h-full rounded-full ' + (r.bestSim >= CFG.MIN_MATCH ? 'bg-lagoon' : 'bg-laterite');
    $('resNote').textContent = note;

    const alts = r.ranked.slice(1, 4);
    $('altWrap').hidden = alts.length === 0;
    const ul = $('altList');
    ul.textContent = '';
    alts.forEach((a) => {
      const li = document.createElement('li');
      li.textContent = `${a.label}: ${Math.round(a.share * 100)}% of votes`;
      ul.appendChild(li);
    });

    $('speciesInput').value = top.label;
    applyTraits(findSpecies(top.label));
  }

  function currentTraitHint(sp) {
    const parts = [];
    if (sp && sp.sex_markers) parts.push('Sex: ' + sp.sex_markers);
    if (sp && sp.age_markers) parts.push('Age: ' + sp.age_markers);
    return parts.join(' ');
  }

  function applyTraits(sp) {
    const pick = (sel, val, list) => {
      const match = list.find((o) => o.toLowerCase().startsWith(String(val || '').toLowerCase()) && val);
      sel.value = match || list[0];
    };
    if (sp) {
      pick($('selBody'), sp.body_form, OPTIONS.body);
      pick($('selMouth'), sp.mouth_position, OPTIONS.mouth);
      pick($('selTail'), sp.tail_shape, OPTIONS.tail);
      $('selRakers').value = sp.gill_rakers && FEEDING_HINT[sp.gill_rakers] !== undefined ? sp.gill_rakers : 'not-visible';
      $('traitHint').textContent = `Pre-filled from the reference for ${sp.common_name}. Check each one against the fish and change what does not match.`;
    } else {
      $('selBody').selectedIndex = 0;
      $('selMouth').selectedIndex = 0;
      $('selTail').selectedIndex = 0;
      $('selRakers').value = 'not-visible';
      $('traitHint').textContent = 'Fill these in from the fish itself. They are saved with the photo.';
    }
    $('selSex').value = 'Unknown';
    $('selAge').value = 'Unknown';
    $('sexAgeHint').textContent = currentTraitHint(sp)
      || 'Sex: check the urogenital papilla and any breeding colours. Age: compare total length with the size at maturity for the species.';
    updateFeeding();
  }

  function updateFeeding() {
    $('feedingNote').textContent = FEEDING_HINT[$('selRakers').value] || '';
  }

  function readTraits() {
    return {
      body_form: $('selBody').value,
      mouth_position: $('selMouth').value,
      tail_shape: $('selTail').value,
      gill_rakers: $('selRakers').value,
      sex: $('selSex').value,
      age_category: $('selAge').value
    };
  }

  function renderKnowledge() {
    const counts = new Map();
    state.records.filter((r) => !r.flip).forEach((r) => counts.set(r.label, (counts.get(r.label) || 0) + 1));
    const photos = [...counts.values()].reduce((a, b) => a + b, 0);

    $('kbSummary').textContent = photos
      ? `${photos} saved ${photos === 1 ? 'photo' : 'photos'} across ${counts.size} ${counts.size === 1 ? 'species' : 'species'}. Aim for at least 10 photos of each species.`
      : 'Nothing saved yet. Photograph a fish, confirm the species and save it.';

    const ul = $('kbList');
    ul.textContent = '';
    [...counts].sort((a, b) => b[1] - a[1]).forEach(([label, n]) => {
      const li = document.createElement('li');
      li.className = 'flex justify-between gap-3 border-b border-mist/60 py-1';
      const name = document.createElement('span');
      name.textContent = label;
      const num = document.createElement('span');
      num.className = 'tabular-nums text-ink/70';
      num.textContent = n;
      li.append(name, num);
      ul.appendChild(li);
    });
  }

  /* ---------- Learning from feedback ---------- */
  async function saveExample() {
    const label = $('speciesInput').value.trim().replace(/\s+/g, ' ');
    if (!label) { toast('Enter the species name first.'); $('speciesInput').focus(); return; }
    if (!state.embOrig) { toast('Take or choose a photo first.'); return; }
    if (state.saved) { toast('This photo is already saved.'); return; }

    // Use the reference spelling when the typed name matches a known species.
    const sp = findSpecies(label);
    const finalLabel = sp ? speciesLabel(sp) : label;

    const stamp = Date.now();
    const traits = readTraits();
    const recs = [
      { id: `${stamp}-a`, label: finalLabel, embedding: state.embOrig, traits, ts: stamp, flip: false },
      { id: `${stamp}-b`, label: finalLabel, embedding: state.embFlip, traits, ts: stamp, flip: true }
    ];

    try {
      for (const rec of recs) await idbPut(rec);
      state.records.push(...recs);
      state.saved = true;
      $('btnSave').textContent = 'Saved';
      $('btnSave').disabled = true;
      renderKnowledge();
      toast(`Saved as ${splitLabel(finalLabel).common || finalLabel}.`);
      if (navigator.storage && navigator.storage.persist) navigator.storage.persist();
    } catch (err) {
      console.error(err);
      toast('Could not save on this device. Storage may be full or blocked.');
    }
  }

  function exportData() {
    if (!state.records.length) { toast('Nothing to export yet.'); return; }
    const blob = new Blob([JSON.stringify({ app: 'identica', version: 1, records: state.records })], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `identica-data-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  async function importData(file) {
    if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      if (data.app !== 'identica' || !Array.isArray(data.records)) throw new Error('bad format');
      const have = new Set(state.records.map((r) => r.id));
      let added = 0;
      for (const r of data.records) {
        const ok = r && typeof r.id === 'string' && typeof r.label === 'string'
          && Array.isArray(r.embedding) && r.embedding.length > 0 && r.embedding.every(Number.isFinite);
        if (!ok || have.has(r.id)) continue;
        const rec = { id: r.id, label: r.label.slice(0, 200), embedding: r.embedding, traits: r.traits || {}, ts: r.ts || Date.now(), flip: !!r.flip };
        await idbPut(rec);
        state.records.push(rec);
        added++;
      }
      renderKnowledge();
      toast(`Imported ${added} ${added === 1 ? 'record' : 'records'}.`);
    } catch (_) {
      toast('That file is not an Identica export.');
    }
  }

  async function clearData() {
    if (!state.records.length) { toast('Nothing to delete.'); return; }
    if (!confirm('Delete everything Identica has learned on this device? Export first if you want a copy.')) return;
    await idbClear();
    state.records = [];
    renderKnowledge();
    toast('Learned data deleted.');
  }

  /* ---------- Install prompt and service worker ---------- */
  let deferredInstall = null;
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredInstall = e;
    $('installBtn').hidden = false;
  });
  window.addEventListener('appinstalled', () => { $('installBtn').hidden = true; deferredInstall = null; });

  function registerServiceWorker() {
    if ('serviceWorker' in navigator && location.protocol !== 'file:') {
      navigator.serviceWorker.register(CFG.SW_URL).catch(() => { /* sw.js is added in the next step */ });
    }
  }

  /* ---------- Start-up ---------- */
  async function init() {
    fillSelect($('selBody'), OPTIONS.body);
    fillSelect($('selMouth'), OPTIONS.mouth);
    fillSelect($('selTail'), OPTIONS.tail);
    fillSelect($('selRakers'), OPTIONS.rakers);
    fillSelect($('selSex'), OPTIONS.sex);
    fillSelect($('selAge'), OPTIONS.age);

    // Camera: opens the rear camera on phones. Gallery: any saved picture.
    $('btnCamera').addEventListener('click', () => $('inputCamera').click());
    $('btnGallery').addEventListener('click', () => $('inputGallery').click());
    [$('inputCamera'), $('inputGallery')].forEach((inp) => {
      inp.addEventListener('change', () => {
        const f = inp.files && inp.files[0];
        inp.value = '';      // lets the same file be chosen again
        handleFile(f);
      });
    });

    $('selRakers').addEventListener('change', updateFeeding);
    $('speciesInput').addEventListener('change', () => {
      const sp = findSpecies($('speciesInput').value.trim());
      if (sp) applyTraits(sp);
    });
    $('btnSave').addEventListener('click', saveExample);
    $('btnExport').addEventListener('click', exportData);
    $('btnImport').addEventListener('click', () => $('inputImport').click());
    $('inputImport').addEventListener('change', (e) => { const f = e.target.files[0]; e.target.value = ''; importData(f); });
    $('btnClear').addEventListener('click', clearData);
    $('installBtn').addEventListener('click', async () => {
      if (!deferredInstall) return;
      deferredInstall.prompt();
      await deferredInstall.userChoice;
      deferredInstall = null;
      $('installBtn').hidden = true;
    });

    registerServiceWorker();
    await loadReferenceData();
    try { state.records = await idbAll(); } catch (_) { state.records = []; }
    renderKnowledge();

    // Start downloading the model in the background so the first photo is faster.
    ensureModel().catch(() => setStatus(''));
  }

  document.addEventListener('DOMContentLoaded', init);
})();
