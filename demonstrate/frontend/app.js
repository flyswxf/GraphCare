const API_BASE = 'http://localhost:8000';

const $ = (s) => document.querySelector(s);

const uploadSection = $('#upload-section');
const resultSection = $('#result-section');
const dropzone = $('#dropzone');
const fileInput = $('#file-input');
const fileName = $('#file-name');
const uploadBtn = $('#upload-btn');
const backBtn = $('#back-btn');

const gauge = $('#gauge');
const probValue = $('#prob-value');
const recommendList = $('#recommend-list');
const similarBars = $('#similar-bars');
const augmentText = $('#augment-text');
const augmentBtn = $('#augment-btn');

let selectedFile = null;
let sessionId = null;

function setGauge(prob) {
  const deg = Math.max(0, Math.min(360, Math.round(prob * 360)));
  gauge.style.setProperty('--val', `${deg}deg`);
  probValue.textContent = `${Math.round(prob * 100)}%`;
  gauge.animate([{ transform: 'scale(0.98)' }, { transform: 'scale(1)' }], { duration: 200, easing: 'ease' });
}

function renderRecommendations(items) {
  recommendList.innerHTML = '';
  (items || []).forEach((it) => {
    const li = document.createElement('li');
    const t = document.createElement('div');
    t.className = 'title';
    t.textContent = it.measure;
    const r = document.createElement('div');
    r.className = 'reason';
    r.textContent = it.reason || '';
    li.appendChild(t); li.appendChild(r);
    recommendList.appendChild(li);
  });
}

function renderSimilarBars(items) {
  similarBars.innerHTML = '';
  (items || []).forEach((it) => {
    const wrap = document.createElement('div');
    wrap.className = 'bar';
    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = it.measure;
    const track = document.createElement('div'); track.className = 'track';
    const fill = document.createElement('div'); fill.className = 'fill';
    track.appendChild(fill);
    wrap.appendChild(name); wrap.appendChild(track);
    similarBars.appendChild(wrap);
    // animate width
    setTimeout(() => { fill.style.width = `${Math.round(it.frequency * 100)}%`; }, 50);
  });
}

function toggleSections(showResult) {
  if (showResult) {
    uploadSection.classList.add('hidden');
    resultSection.classList.remove('hidden');
  } else {
    resultSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
  }
}

function setLoading(btn, loading) {
  btn.disabled = loading || (!selectedFile && btn === uploadBtn);
  btn.textContent = loading ? '处理中...' : (btn === uploadBtn ? '上传并推理' : '提交补充信息并更新');
}

async function inferByUpload(file) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(`${API_BASE}/api/infer/upload`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(`后端错误：${res.status}`);
  return res.json();
}

async function augment(text) {
  const res = await fetch(`${API_BASE}/api/infer/augment`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, text })
  });
  if (!res.ok) throw new Error(`后端错误：${res.status}`);
  return res.json();
}

function wireDropzone() {
  const prevent = (e) => { e.preventDefault(); e.stopPropagation(); };
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((ev) => dropzone.addEventListener(ev, prevent));
  dropzone.addEventListener('dragover', () => dropzone.classList.add('hover'));
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('hover'));
  dropzone.addEventListener('drop', (e) => {
    dropzone.classList.remove('hover');
    const f = e.dataTransfer.files[0];
    if (f) { selectedFile = f; fileName.textContent = f.name; uploadBtn.disabled = false; }
  });
  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    const f = e.target.files[0];
    if (f) { selectedFile = f; fileName.textContent = f.name; uploadBtn.disabled = false; }
  });
}

function init() {
  wireDropzone();

  uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    try {
      setLoading(uploadBtn, true);
      const data = await inferByUpload(selectedFile);
      sessionId = data.session_id;
      setGauge(data.probability);
      renderRecommendations(data.recommended);
      renderSimilarBars(data.similar_cases);
      toggleSections(true);
    } catch (err) {
      alert(err.message || '推理失败');
    } finally {
      setLoading(uploadBtn, false);
    }
  });

  augmentBtn.addEventListener('click', async () => {
    const text = augmentText.value.trim();
    if (!text) { augmentText.focus(); return; }
    try {
      setLoading(augmentBtn, true);
      const data = await augment(text);
      setGauge(data.probability);
      renderRecommendations(data.recommended);
      renderSimilarBars(data.similar_cases);
      augmentText.value = '';
    } catch (err) {
      alert(err.message || '更新失败');
    } finally {
      setLoading(augmentBtn, false);
    }
  });

  backBtn.addEventListener('click', () => {
    // reset state
    selectedFile = null; sessionId = null; fileName.textContent = '未选择文件'; uploadBtn.disabled = true;
    recommendList.innerHTML = ''; similarBars.innerHTML = ''; setGauge(0);
    toggleSections(false);
  });

  // init gauge
  setGauge(0);
}

document.addEventListener('DOMContentLoaded', init);