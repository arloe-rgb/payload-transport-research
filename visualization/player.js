const byId = (id) => document.getElementById(id);
const palette = ["#168075", "#be5428", "#5c6dc5", "#a54b86", "#647c28", "#1677a2"];
const color = (index) => palette[index % palette.length];
const scene = byId("scene");
const timeline = byId("timeline");
const chartRoot = byId("charts");
let data, time, index, playing = false, lastFrame, frameRequest;
let charts = [];
let scene3D;

function canvasSize(canvas) {
  const {width, height} = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(width * ratio)) canvas.width = Math.round(width * ratio);
  if (canvas.height !== Math.round(height * ratio)) canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return {ctx, width, height};
}

function validate(replay) {
  const {payload, time: samples, forceLimits} = replay;
  const vector = (row, size) => Array.isArray(row) && row.length === size && row.every(Number.isFinite);
  if (replay.version !== 1 || !Array.isArray(samples) || !samples.length ||
      !samples.every((t, i) => Number.isFinite(t) && (!i || t > samples[i - 1]))) {
    throw new Error("Expected a version 1 replay with increasing timestamps.");
  }
  const count = payload?.attachments?.length;
  if (!(count > 0) || ![payload.width, payload.height].every((n) => Number.isFinite(n) && n > 0) ||
      !Array.isArray(payload.attachments) || !payload.attachments.every((p) => vector(p, 2)) ||
      !vector(forceLimits, count) || !forceLimits.every((n) => n > 0)) {
    throw new Error("Payload dimensions, attachment points, or force limits are invalid.");
  }
  for (const [key, size] of Object.entries({
    state: 6, reference: 6, u: 2 * count, lambda: 3, gradientU: 2 * count, gradientLambda: 3,
  })) {
    if (!Array.isArray(replay[key]) || replay[key].length !== samples.length ||
        !replay[key].every((row) => vector(row, size))) {
      throw new Error(`Invalid or missing replay history: ${key}.`);
    }
  }
}

function locate(t) {
  let lo = 0;
  let hi = data.time.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (data.time[mid] <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

function interpolate(rows) {
  const next = Math.min(index + 1, rows.length - 1);
  if (next === index) return rows[index];
  const fraction = (time - data.time[index]) / (data.time[next] - data.time[index]);
  return rows[index].map((value, j) => value + fraction * (rows[next][j] - value));
}

function seek(t) {
  time = Math.max(data.time[0], Math.min(data.time.at(-1), t));
  index = locate(time);
  timeline.value = index;
  timeline.setAttribute("aria-valuetext", `${time.toFixed(2)} seconds`);
  byId("clock").textContent = `${time.toFixed(2)} s`;
  drawScene();
  for (const chart of charts) updateCursor(chart);
}

function pause() {
  playing = false;
  cancelAnimationFrame(frameRequest);
  lastFrame = undefined;
  byId("play").textContent = "Play";
}

function frame(now) {
  if (!playing) return;
  if (lastFrame !== undefined) seek(time + (now - lastFrame) / 1000 * Number(byId("speed").value));
  lastFrame = now;
  if (time >= data.time.at(-1)) pause();
  else frameRequest = requestAnimationFrame(frame);
}

function drawScene() {
  if (!data) return;
  const pose = interpolate(data.state);
  const [x, z, theta] = pose;
  byId("pose-x").textContent = `${x.toFixed(2)} m`;
  byId("pose-z").textContent = `${z.toFixed(2)} m`;
  byId("pose-theta").textContent = `${theta.toFixed(3)} rad`;
  if (!scene3D) return;

  const setting = (id, fallback) => {
    const input = byId(id);
    return input.validity.valid && input.valueAsNumber > 0 ? input.valueAsNumber : fallback;
  };
  const next = Math.min(index + 1, data.time.length - 1);
  const fraction = next === index ? 0 : (time - data.time[index]) / (data.time[next] - data.time[index]);
  const visible = Object.fromEntries(["reference", "actual", "forces", "cables", "labels"].map((layer) =>
    [layer, byId(`show-${layer}`).checked]));
  scene3D.update({
    pose, index, fraction, time, visible,
    reference: interpolate(data.reference), forces: interpolate(data.u),
    cableLength: setting("cable-length", 1.6), arrowScale: setting("force-scale", 0.18),
  });
}

function updateCursor(chart) {
  if (!chart.width) return;
  const elapsed = data.time.at(-1) - data.time[0];
  const fraction = elapsed ? (time - data.time[0]) / elapsed : 0;
  chart.cursor.style.left = `${chart.left + fraction * chart.width}px`;
}

function renderChart(chart) {
  const {ctx, width, height} = canvasSize(chart.canvas);
  if (width < 80 || height < 40) return;
  const left = 49, right = width - 13, top = 10, bottom = height - 25;
  chart.left = left;
  chart.width = right - left;
  const t0 = data.time[0], t1 = data.time.at(-1);
  const xAt = (t) => left + (t - t0) / (t1 - t0 || 1) * chart.width;
  let min = Infinity, max = -Infinity;
  for (const {values} of chart.series) {
    for (const value of values) {
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
  }
  const padding = max === min ? Math.max(Math.abs(max) * 0.1, 0.1) : (max - min) * 0.07;
  min -= padding;
  max += padding;
  const yAt = (value) => bottom - (value - min) / (max - min) * (bottom - top);
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillStyle = "#536a72";
  ctx.strokeStyle = "#d9e3e0";
  ctx.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick++) {
    const value = min + tick / 4 * (max - min);
    const y = yAt(value);
    const x = left + tick / 4 * chart.width;
    const t = t0 + tick / 4 * (t1 - t0);
    ctx.beginPath();
    ctx.moveTo(left, y); ctx.lineTo(right, y);
    ctx.moveTo(x, top); ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(Number(value.toPrecision(3)).toString(), left - 5, y + 3);
    ctx.textAlign = tick === 4 ? "right" : tick === 0 ? "left" : "center";
    ctx.fillText(`${Number(t.toFixed(2))} s`, x, bottom + 16);
  }
  for (const {values, tint, dashed} of chart.series) {
    ctx.strokeStyle = tint;
    ctx.lineWidth = 1.65;
    ctx.setLineDash(dashed ? [5, 4] : []);
    ctx.beginPath();
    values.forEach((value, i) => {
      const x = xAt(data.time[i]), y = yAt(value);
      if (i) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
  updateCursor(chart);
}

const chartObserver = new ResizeObserver((entries) => {
  for (const {target} of entries) {
    const chart = charts.find(({canvas}) => canvas === target);
    if (chart) renderChart(chart);
  }
});

function buildCharts() {
  chartObserver.disconnect();
  chartRoot.replaceChildren();
  charts = [];
  const component = (rows, i) => rows.map((row) => row[i]);
  const series = (label, values, tint, dashed = false) => ({label, values, tint, dashed});
  const perUav = (makeSeries) => data.forceLimits.map((limit, i) => makeSeries(i, limit)).flat();
  const forces = (axis) => perUav((i) => series(`UAV ${i + 1}`, component(data.u, 2 * i + axis), color(i)));
  const track = ["x (m)", "z (m)", "θ (rad)"].map((label, i) => [label, [
    series("Actual", component(data.state, i), palette[0]),
    series("Reference", component(data.reference, i), palette[1], true),
  ]]);
  const norm = (key) => data[key].map((row) => Math.hypot(...row));
  const groups = [
    ["Payload tracking", track],
    ["Control allocation", [
      ["Force magnitude (N)", perUav((i, limit) => [
        series(`UAV ${i + 1}`, data.u.map((row) => Math.hypot(row[2 * i], row[2 * i + 1])), color(i)),
        series(`UAV ${i + 1} limit`, data.time.map(() => limit), color(i), true),
      ])],
      ["Horizontal force Fx (N)", forces(0)],
      ["Vertical force Fz (N)", forces(1)],
    ]],
    ["Optimization dynamics", [
      ["Decision variables u (N)", perUav((i) => [
        series(`UAV ${i + 1} x`, component(data.u, 2 * i), color(i)),
        series(`UAV ${i + 1} z`, component(data.u, 2 * i + 1), color(i), true),
      ])],
      ["Equality multipliers λ", ["Force x", "Force z", "Pitch torque"].map((label, i) =>
        series(label, component(data.lambda, i), color(i)))],
      ["Gradient norm ‖∇ᵤℒ‖₂", [series("Control gradient", norm("gradientU"), palette[0])]],
      ["Gradient norm ‖∇λℒ‖₂", [series("Multiplier gradient", norm("gradientLambda"), palette[1])]],
    ]],
  ];

  for (const [name, plots] of groups) {
    const group = document.createElement("section");
    group.className = "chart-group";
    group.innerHTML = "<h2></h2><div class='chart-grid'></div>";
    group.firstChild.textContent = name;
    chartRoot.append(group);
    for (const [label, traces] of plots) {
      const card = document.createElement("article");
      card.className = "chart";
      card.innerHTML = "<h3></h3><div class='chart-area'><canvas role='img'></canvas><span class='cursor'></span></div><div class='chart-key'></div>";
      card.firstChild.textContent = label;
      const canvas = card.querySelector("canvas");
      canvas.setAttribute("aria-label", `${label} over simulation time`);
      const chart = {canvas, cursor: card.querySelector(".cursor"), series: traces};
      for (const trace of traces) {
        const key = document.createElement("span");
        const swatch = document.createElement("i");
        swatch.style.color = trace.tint;
        swatch.style.borderTopStyle = trace.dashed ? "dashed" : "solid";
        key.append(swatch, trace.label);
        card.querySelector(".chart-key").append(key);
      }
      canvas.addEventListener("click", (event) => {
        if (!chart.width) return;
        const fraction = (event.clientX - canvas.getBoundingClientRect().left - chart.left) / chart.width;
        pause();
        seek(data.time[0] + fraction * (data.time.at(-1) - data.time[0]));
      });
      group.lastChild.append(card);
      charts.push(chart);
      chartObserver.observe(canvas);
    }
  }
}

function load(replay, name) {
  validate(replay);
  pause();
  data = replay;
  scene3D?.load(data);
  byId("empty").hidden = true;
  byId("run-info").textContent = `${data.forceLimits.length} UAVs · ${data.time.length.toLocaleString()} samples`;
  byId("run-info").title = name;
  byId("duration").textContent = `${data.time.at(-1).toFixed(2)} s`;
  timeline.max = data.time.length - 1;
  byId("playback").disabled = false;
  buildCharts();
  seek(data.time[0]);
}

timeline.addEventListener("input", () => { pause(); seek(data.time[Number(timeline.value)]); });
byId("play").addEventListener("click", () => {
  if (playing) return pause();
  if (time >= data.time.at(-1)) seek(data.time[0]);
  playing = true;
  byId("play").textContent = "Pause";
  frameRequest = requestAnimationFrame(frame);
});
byId("restart").addEventListener("click", () => { pause(); seek(data.time[0]); });
for (const [id, delta] of [["back", -1], ["forward", 1]]) {
  byId(id).addEventListener("click", () => {
    pause();
    seek(data.time[Math.max(0, Math.min(data.time.length - 1, index + delta))]);
  });
}
document.querySelectorAll(".layers input").forEach((input) => input.addEventListener("input", drawScene));
try {
  scene3D = new ReplayScene(scene);
} catch (error) {
  console.warn("3D scene unavailable", error);
  byId("scene-error").hidden = false;
  byId("camera-controls").disabled = true;
}
byId("camera-view").addEventListener("change", (event) => scene3D?.setView(event.target.value));
byId("camera-reset").addEventListener("click", () => scene3D?.setView(byId("camera-view").value));
byId("camera-follow").addEventListener("change", (event) => scene3D?.setFollow(event.target.checked));

byId("file").addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;
  try {
    load(JSON.parse(await file.text()), file.name);
  } catch (error) {
    byId("run-info").textContent = error.message;
  } finally {
    event.target.value = "";
  }
});

if (location.protocol === "file:") showEmpty();
else fetch("replay.json", {cache: "no-store"}).then(async (response) => {
  if (data) return;
  if (!response.ok) return showEmpty();
  const replay = await response.json();
  if (!data) load(replay, "replay.json");
}).catch((error) => {
  if (!data) showEmpty(error.message);
});

function showEmpty(message = "Open an exported replay JSON to begin.") {
  byId("run-info").textContent = "No replay loaded";
  byId("empty").textContent = message;
  byId("empty").hidden = false;
}
