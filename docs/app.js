// Form Coach demo: camera / uploaded video / recorded skeleton → MediaPipe pose → exercise, reps and form cues.
// Everything runs locally in the browser; nothing is uploaded anywhere.
import { FilesetResolver, PoseLandmarker } from "./vendor/mediapipe/vision_bundle.mjs";
import { RepCounter, checkRep, classify, frameFeatures, primarySignal } from "./coach.js";

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");
const NAMES = { squat: "Squat", push_up: "Push-up", bicep_curl: "Biceps curl", shoulder_press: "Shoulder press" };
const BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [27, 31], [28, 32]];
const FPS = 10; // features are sampled at 10 fps, like the training data

let model = null, thresholds = {}, landmarker = null, stream = null, running = false, replay = null, lastTs = 0;
let state;
// MediaPipe needs strictly increasing timestamps for the whole life of the landmarker (across sessions too)
const nextTs = () => (lastTs = Math.max(lastTs + 1, Math.round(performance.now())));

function reset() {
  state = { frames: [], lastT: -1, probs: null, exercise: null, counter: null, repStart: 0, reps: 0 };
  $("reps").textContent = "0";
  $("exercise").textContent = "—";
  $("confidence").textContent = "detected automatically";
  $("phase").style.width = "0%";
  const list = $("cues");
  list.replaceChildren(Object.assign(document.createElement("li"), { className: "muted", textContent: "Each finished rep gets checked here." }));
}

const status = (msg) => { $("status").textContent = msg; };

async function loadModels() {
  if (!model) {
    model = await (await fetch("model.json")).json();
    thresholds = (await (await fetch("form_thresholds.json")).json()).thresholds;
  }
  if (!landmarker) {
    status("Loading the pose model…");
    const files = await FilesetResolver.forVisionTasks("vendor/mediapipe/wasm");
    landmarker = await PoseLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: "models/pose_landmarker_lite.task", delegate: "GPU" },
      runningMode: "VIDEO", numPoses: 1,
    });
  }
}

function currentExercise() {
  return $("override").value || state.exercise;
}

/** One pose (world landmarks) at time t (seconds) → update exercise, reps and cues. */
function step(world, t) {
  if (t - state.lastT < 1 / FPS - 1e-3) return; // keep 10 fps
  state.lastT = t;
  const f = frameFeatures(world);
  state.frames.push(f);
  if (state.frames.length > 600) { state.frames.shift(); state.repStart = Math.max(0, state.repStart - 1); }

  // exercise: smoothed probabilities over 2-second windows, every half second
  if (state.frames.length >= model.window && state.frames.length % 5 === 0) {
    const p = classify(model, state.frames.slice(-model.window));
    state.probs = state.probs ? state.probs.map((v, i) => 0.7 * v + 0.3 * p[i]) : p;
    const best = state.probs.indexOf(Math.max(...state.probs));
    if (!state.exercise || state.probs[best] > 0.75) state.exercise = model.classes[best];
    $("exercise").textContent = NAMES[currentExercise()];
    $("confidence").textContent = $("override").value ? "chosen by you" : `${Math.round(state.probs[best] * 100)}% sure`;
  }
  const ex = currentExercise();
  if (!ex) return;
  if (!state.counter || state.counter.exercise !== ex) {
    state.counter = Object.assign(new RepCounter(), { exercise: ex });
    state.repStart = state.frames.length - 1;
  }
  const c = state.counter;
  const done = c.push(primarySignal(ex, f), t);
  const vals = c.hist.map(([, v]) => v), now = vals[vals.length - 1];
  const lo = Math.min(...vals), hi = Math.max(...vals);
  $("phase").style.width = `${hi > lo ? Math.round(((now - lo) / (hi - lo)) * 100) : 0}%`;
  if (done) {
    state.reps += 1;
    $("reps").textContent = String(state.reps);
    const rep = state.frames.slice(state.repStart);
    state.repStart = state.frames.length - 1;
    addCues(state.reps, checkRep(ex, rep, thresholds));
  }
}

function addCues(n, cues) {
  const list = $("cues");
  if (list.firstElementChild?.classList.contains("muted")) list.replaceChildren();
  const li = document.createElement("li");
  li.className = cues.length ? "fix" : "good";
  li.textContent = `Rep ${n}: ${cues.length ? cues.join(" · ") : "good form ✓"}`; // textContent: no HTML injection
  list.prepend(li);
}

function drawSkeleton(points, w, h, map) {
  ctx.lineWidth = Math.max(3, w / 220);
  ctx.strokeStyle = "#ff7a1a";
  ctx.fillStyle = "#ffd166";
  for (const [a, b] of BONES) {
    const [x1, y1] = map(points[a]), [x2, y2] = map(points[b]);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  for (const i of [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) {
    const [x, y] = map(points[i]);
    ctx.beginPath(); ctx.arc(x, y, ctx.lineWidth * 1.2, 0, Math.PI * 2); ctx.fill();
  }
}

/** Detect the pose in the current video frame, draw it and feed it to the coach. */
function analyseFrame(t) {
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);
  const res = landmarker.detectForVideo(video, nextTs());
  if (!res.worldLandmarks?.length) return false;
  step(res.worldLandmarks[0].map((p) => [p.x, p.y, p.z]), t);
  drawSkeleton(res.landmarks[0], canvas.width, canvas.height, (p) => [p.x * canvas.width, p.y * canvas.height]);
  return true;
}

/** Camera: real time, as fast as the device allows (the counter works on timestamps, not frame counts). */
function loopCamera() {
  if (!running) return;
  if (video.readyState >= 2) {
    status(analyseFrame(performance.now() / 1000) ? "Tracking… (camera)" : "Step back so your whole body is in view.");
  }
  requestAnimationFrame(loopCamera);
}

/** Uploaded video: step through it at exactly 10 frames per second — same as the evaluation, on any device. */
async function analyseFile() {
  video.pause();
  const seek = (t) => new Promise((ok) => { video.addEventListener("seeked", ok, { once: true }); video.currentTime = t; });
  for (let t = 0; running && t < video.duration; t += 1 / FPS) {
    await seek(t);
    const found = analyseFrame(t);
    status(`Analysing your video… ${Math.round((t / video.duration) * 100)} %${found ? "" : " — no full body in view"}`);
  }
  if (running) { status(`Done — ${state.reps} reps.`); stop(); }
}

function loopReplay() {
  if (!running || !replay) return;
  const i = Math.floor(((performance.now() - replay.start) / 1000) * replay.fps);
  if (i >= replay.frames.length) { status(`Replay finished — ${state.reps} reps counted.`); stop(); return; }
  const world = replay.frames[i], t = i / replay.fps;
  step(world, t);
  canvas.width = 960; canvas.height = 600;
  ctx.fillStyle = "#07090c"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  const scale = canvas.height / 1.7; // world landmarks are metres around the hips
  drawSkeleton(world, canvas.width, canvas.height, (p) => [canvas.width / 2 + p[0] * scale, canvas.height * 0.5 + p[1] * scale]);
  status(`Replaying a recorded set (skeleton only) — ${Math.floor(t)} s`);
  requestAnimationFrame(loopReplay);
}

function stop() {
  running = false;
  stream?.getTracks().forEach((tr) => tr.stop());
  stream = null; replay = null;
  video.pause();
  if (video.src.startsWith("blob:")) URL.revokeObjectURL(video.src);
  video.removeAttribute("src"); video.srcObject = null;
  $("btn-stop").disabled = true;
}

async function start(kind, payload) {
  stop(); reset();
  try {
    await loadModels();
  } catch (e) {
    status("Could not load the AI model in this browser."); console.error(e); return;
  }
  running = true; $("btn-stop").disabled = false;
  if (kind === "replay") {
    status("Loading the recorded set…");
    replay = await (await fetch(`samples/${payload}.json`)).json();
    replay.start = performance.now();
    requestAnimationFrame(loopReplay);
    return;
  }
  if (kind === "camera") {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 } }, audio: false });
    } catch {
      status("Camera permission was denied or no camera was found."); stop(); return;
    }
    video.srcObject = stream;
    await video.play();
    requestAnimationFrame(loopCamera);
  } else {
    video.src = URL.createObjectURL(payload); // stays in the browser — never uploaded
    await new Promise((ok, fail) => { video.onloadedmetadata = ok; video.onerror = fail; }).catch(() => null);
    if (!video.duration) { status("This video format can't be read by your browser."); stop(); return; }
    analyseFile();
  }
}

$("btn-camera").addEventListener("click", () => start("camera"));
$("file").addEventListener("change", (e) => { const f = e.target.files?.[0]; if (f) start("file", f); e.target.value = ""; });
$("sample").addEventListener("change", (e) => { if (e.target.value) start("replay", e.target.value); e.target.value = ""; });
$("btn-stop").addEventListener("click", () => { stop(); status("Stopped."); });
$("override").addEventListener("change", () => { if (state) { state.counter = null; $("exercise").textContent = NAMES[currentExercise()] || "—"; } });
reset();
