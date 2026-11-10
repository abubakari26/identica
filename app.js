/* Identica - pure TensorFlow.js version (no teachablemachine library needed)
 * Works with a Teachable Machine image model exported as "Tensorflow.js".
 * Put model.json, metadata.json and the weights (.bin) files in ./my_model/
 */

const MODEL_URL = "./my_model/model.json";
const METADATA_URL = "./my_model/metadata.json";
const INPUT_SIZE = 224; // Teachable Machine image models use 224x224

let model = null;
let labels = [];
let stream = null;
let liveLoopRunning = false;

const els = {
  status: document.getElementById("status"),
  video: document.getElementById("video"),
  preview: document.getElementById("preview"),
  results: document.getElementById("results"),
  fileInput: document.getElementById("fileInput"),
  startCamBtn: document.getElementById("startCamBtn"),
  stopCamBtn: document.getElementById("stopCamBtn"),
  snapBtn: document.getElementById("snapBtn"),
};

function setStatus(msg) {
  els.status.textContent = msg;
}

// 1) Load the model and class labels
async function loadModel() {
  if (location.protocol === "file:") {
    setStatus("ERROR: You opened this file directly. Run it from a web server (python -m http.server) or GitHub Pages, then open http://localhost:8000");
    return;
  }
  if (typeof tf === "undefined") {
    setStatus("ERROR: TensorFlow.js did not load. Check your internet connection.");
    return;
  }
  try {
    setStatus("Loading model...");
    const check = await fetch(MODEL_URL, { method: "HEAD" });
    if (!check.ok) {
      setStatus("ERROR: ./my_model/model.json not found (" + check.status + "). Export your Teachable Machine model and put its files in a folder named my_model next to index.html.");
      return;
    }
    model = await tf.loadLayersModel(MODEL_URL);
    const meta = await (await fetch(METADATA_URL)).json();
    labels = meta.labels;
    tf.tidy(() => model.predict(tf.zeros([1, INPUT_SIZE, INPUT_SIZE, 3])));
    setStatus("Model ready (" + labels.length + " classes). Take or upload a photo.");
  } catch (err) {
    console.error(err);
    setStatus("ERROR loading model: " + err.message);
  }
}

// 2) Turn an image/video/canvas into a model-ready tensor
function preprocess(source) {
  return tf.tidy(() => {
    let t = tf.browser.fromPixels(source).toFloat();
    t = tf.image.resizeBilinear(t, [INPUT_SIZE, INPUT_SIZE]);
    t = t.div(127.5).sub(1); // scale to [-1, 1] like Teachable Machine
    return t.expandDims(0);
  });
}

// 3) Run inference, return sorted [{label, probability}]
async function predict(source) {
  if (!model) {
    setStatus("Model not loaded yet.");
    return [];
  }
  const input = preprocess(source);
  const output = model.predict(input);
  const probs = await output.data();
  input.dispose();
  output.dispose();
  return Array.from(probs)
    .map((p, i) => ({ label: labels[i], probability: p }))
    .sort((a, b) => b.probability - a.probability);
}

function showResults(preds) {
  els.results.innerHTML = "";
  preds.forEach((p, i) => {
    const row = document.createElement("div");
    row.className = "row" + (i === 0 ? " top" : "");
    const pct = (p.probability * 100).toFixed(1);
    row.innerHTML =
      '<span class="name"></span><span class="pct">' + pct + '%</span>' +
      '<div class="bar"><div style="width:' + pct + '%"></div></div>';
    row.querySelector(".name").textContent = p.label;
    els.results.appendChild(row);
  });
}

// 4) Upload or take a photo (phone camera via capture attribute)
els.fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  els.preview.onload = async () => {
    showResults(await predict(els.preview));
    URL.revokeObjectURL(url);
  };
  els.preview.src = url;
  els.preview.style.display = "block";
});

// 5) Live webcam mode (like the Teachable Machine sample)
els.startCamBtn.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    els.video.srcObject = stream;
    await els.video.play();
    els.video.style.display = "block";
    els.startCamBtn.disabled = true;
    els.stopCamBtn.disabled = false;
    els.snapBtn.disabled = false;
    liveLoopRunning = true;
    liveLoop();
  } catch (err) {
    console.error(err);
    setStatus("Camera access denied or unavailable.");
  }
});

els.stopCamBtn.addEventListener("click", stopCamera);

function stopCamera() {
  liveLoopRunning = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  els.video.style.display = "none";
  els.startCamBtn.disabled = false;
  els.stopCamBtn.disabled = true;
  els.snapBtn.disabled = true;
}

async function liveLoop() {
  if (!liveLoopRunning) return;
  showResults(await predict(els.video));
  setTimeout(() => requestAnimationFrame(liveLoop), 250); // ~4 predictions/sec saves battery
}

// Freeze the current webcam frame into the preview
els.snapBtn.addEventListener("click", async () => {
  const c = document.createElement("canvas");
  c.width = els.video.videoWidth;
  c.height = els.video.videoHeight;
  c.getContext("2d").drawImage(els.video, 0, 0);
  els.preview.src = c.toDataURL("image/jpeg", 0.9);
  els.preview.style.display = "block";
  stopCamera();
  els.preview.onload = async () => showResults(await predict(els.preview));
});

loadModel();
