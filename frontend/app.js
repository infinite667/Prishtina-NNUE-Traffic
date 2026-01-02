/* global L, Chart */

const state = {
  map: null,
  roadsLayer: null,
  lights: new Map(),      // id -> marker
  vehicles: new Map(),    // id -> marker
  lastVehiclePayload: new Map(),
  lastLightPayload: new Map(),
  selection: { type: null, id: null },
  glowCircle: null,
  ws: null,
  settings: { nnue_enabled: false, speed_multiplier: 1.0, target_vehicle_count: 500 },
  chart: null,
  chartData: { labels: [], values: [] },
  lastMetricPushT: 0
};

function svgDataUrl(svgText, colorHex) {
  const colored = svgText.replaceAll("CURRENT", colorHex);
  const encoded = encodeURIComponent(colored)
    .replaceAll("'", "%27")
    .replaceAll('"', "%22");
  return `data:image/svg+xml,${encoded}`;
}

async function loadText(url) {
  const res = await fetch(url);
  return res.text();
}

function makeCarIcon(color) {
  const svg = window.__CAR_SVG__;
  return L.icon({
    iconUrl: svgDataUrl(svg, color),
    iconSize: [28, 28],
    iconAnchor: [14, 14]
  });
}

function makeLightIcon() {
  // Backwards compat (shouldn't be called anymore)
  return makeLightIconForState("GREEN");
}

function replaceFirst(haystack, needle, replacement) {
  const idx = haystack.indexOf(needle);
  if (idx === -1) return haystack;
  return haystack.slice(0, idx) + replacement + haystack.slice(idx + needle.length);
}

const lightIconCache = new Map(); // state -> L.icon
function makeLightIconForState(lightState) {
  const key = String(lightState || "GREEN").toUpperCase();
  if (lightIconCache.has(key)) return lightIconCache.get(key);

  const template = window.__LIGHT_SVG__;
  // Template has 3 circles in order: red, yellow, green.
  const dim = "#334155";
  let svg = template;
  svg = replaceFirst(svg, 'fill="#ef4444"', `fill="${key === "RED" ? "#ef4444" : dim}"`);
  svg = replaceFirst(svg, 'fill="#f59e0b"', `fill="${key === "YELLOW" ? "#f59e0b" : dim}"`);
  svg = replaceFirst(svg, 'fill="#22c55e"', `fill="${key === "GREEN" ? "#22c55e" : dim}"`);

  const icon = L.icon({
    iconUrl: svgDataUrl(svg, "#ffffff"),
    iconSize: [30, 30],
    iconAnchor: [15, 15]
  });
  lightIconCache.set(key, icon);
  return icon;
}

function initMap() {
  const m = L.map("map", {
    zoomControl: true,
    preferCanvas: true
  }).setView([42.6629, 21.1655], 13);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors"
  }).addTo(m);

  state.map = m;
  state.roadsLayer = L.layerGroup().addTo(m);
}

function drawRoads(roads) {
  // Roads are kept for routing and bounds, but we don't render them to keep vehicles clearly visible.
  state.roadsLayer.clearLayers();
}

function ensureGlowCircle(latlng) {
  if (!state.glowCircle) {
    state.glowCircle = L.circle(latlng, {
      radius: 35,
      weight: 2,
      opacity: 0.9,
      fillOpacity: 0.05,
      className: "glow-ring"
    }).addTo(state.map);
  } else {
    state.glowCircle.setLatLng(latlng);
  }
}

function setSelection(type, id) {
  state.selection = { type, id };
  if (!type) {
    document.getElementById("selectionEmpty").classList.remove("hidden");
    document.getElementById("selectionDetails").classList.add("hidden");
    return;
  }
  document.getElementById("selectionEmpty").classList.add("hidden");
  document.getElementById("selectionDetails").classList.remove("hidden");
}

function renderSelectionDetails(payload) {
  const box = document.getElementById("selectionDetails");
  if (!payload) {
    box.innerHTML = `<div class="muted">No data.</div>`;
    return;
  }
  if (state.selection.type === "vehicle") {
    const rerouted = payload.rerouted;
    const badge = rerouted
      ? `<span class="badge green">NNUE: re-routing</span>`
      : `<span class="badge blue">Main route</span>`;
    box.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
        <div style="font-weight:800;">Vehicle #${payload.id}</div>
        ${badge}
      </div>
      <div class="details-row"><div class="details-key">Speed</div><div>${payload.speed_kmh} km/h</div></div>
      <div class="details-row"><div class="details-key">Distance traveled</div><div>${payload.distance_traveled_m} m</div></div>
      <div class="details-row"><div class="details-key">Distance to destination</div><div>${payload.distance_to_dest_m} m</div></div>
      <div class="details-row"><div class="details-key">Time in simulation</div><div>${payload.time_in_sim_s} s</div></div>
      <div class="details-row"><div class="details-key">Waiting time</div><div>${payload.waiting_time_s} s</div></div>
      <div class="details-row"><div class="details-key">State</div><div>${payload.state}</div></div>
    `;
  } else if (state.selection.type === "light") {
    box.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div style="font-weight:800;">Traffic Light</div>
        <span class="badge blue">ID: ${payload.id}</span>
      </div>
      <div class="details-row"><div class="details-key">State</div><div>${payload.state}</div></div>
      <div class="details-row"><div class="details-key">Lat</div><div>${payload.lat.toFixed(6)}</div></div>
      <div class="details-row"><div class="details-key">Lon</div><div>${payload.lon.toFixed(6)}</div></div>
    `;
  }
}

function upsertLight(light) {
  const id = light.id;
  state.lastLightPayload.set(id, light);
  const marker = state.lights.get(id);
  if (!marker) {
    const m = L.marker([light.lat, light.lon], { icon: makeLightIconForState(light.state) })
      .addTo(state.map)
      .on("click", () => {
        setSelection("light", id);
        ensureGlowCircle(m.getLatLng());
        renderSelectionDetails(state.lastLightPayload.get(id));
      });
    state.lights.set(id, m);
  } else {
    marker.setLatLng([light.lat, light.lon]);
    marker.setIcon(makeLightIconForState(light.state));
  }
}

function colorForVehicleState(v) {
  if (v.state === "ARRIVED") return "#a78bfa";
  if (v.state === "WAITING") return "#fb7185";
  return "#38bdf8";
}

function upsertVehicle(v) {
  const id = v.id;
  const existing = state.vehicles.get(id);
  const color = colorForVehicleState(v);
  if (!existing) {
    const marker = L.marker([v.lat, v.lon], { icon: makeCarIcon(color) })
      .addTo(state.map)
      .on("click", () => {
        setSelection("vehicle", id);
        ensureGlowCircle(marker.getLatLng());
        renderSelectionDetails(v);
      });
    state.vehicles.set(id, marker);
  } else {
    // Update icon only when the logical state changes (prevents DOM churn + flicker)
    const prev = state.lastVehiclePayload.get(id);
    if (!prev || prev.state !== v.state) {
      existing.setIcon(makeCarIcon(color));
    }
    // LatLng update handled in animation loop
  }
  state.lastVehiclePayload.set(id, v);
}

function pruneVehicles(currentIds) {
  for (const id of Array.from(state.vehicles.keys())) {
    if (!currentIds.has(id)) {
      state.map.removeLayer(state.vehicles.get(id));
      state.vehicles.delete(id);
      state.lastVehiclePayload.delete(id);
    }
  }
}

function initChart() {
  const ctx = document.getElementById("waitChart");
  state.chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Avg waiting (NNUE off)",
          data: [],
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25,
          borderColor: "#ef4444" // red
        },
        {
          label: "Avg waiting (NNUE on)",
          data: [],
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25,
          borderColor: "#22c55e" // green
        }
      ]
    },
    options: {
      responsive: true,
      animation: false,
      plugins: {
        legend: { display: true }
      },
      scales: {
        x: { display: false },
        y: {
          ticks: { color: "rgba(148,163,184,.9)" },
          grid: { color: "rgba(255,255,255,.06)" }
        }
      }
    }
  });
}

function pushMetric(t, avgWait, nnueEnabled) {
  // 1 point per second
  state.chart.data.labels.push(t.toFixed(0));
  // Keep two distinct series with gaps so it's easy to compare ON vs OFF.
  state.chart.data.datasets[0].data.push(nnueEnabled ? null : avgWait);
  state.chart.data.datasets[1].data.push(nnueEnabled ? avgWait : null);
  if (state.chart.data.labels.length > 180) {
    state.chart.data.labels.shift();
    state.chart.data.datasets[0].data.shift();
    state.chart.data.datasets[1].data.shift();
  }
  state.chart.update("none");
}

function applySettingsUI() {
  const pressed = state.settings.nnue_enabled;
  const toggle = document.getElementById("nnueToggle");
  toggle.setAttribute("aria-pressed", pressed ? "true" : "false");
  const slider = document.getElementById("speedSlider");
  slider.value = state.settings.speed_multiplier;
  document.getElementById("speedValue").innerText = Number(state.settings.speed_multiplier).toFixed(1);

  const mapping = [200, 500, 1000, 3000];
  const vcSlider = document.getElementById("vehicleCountSlider");
  const idx = Math.max(0, mapping.indexOf(state.settings.target_vehicle_count));
  vcSlider.value = String(idx === -1 ? 1 : idx);
  document.getElementById("vehicleCountValue").innerText = String(state.settings.target_vehicle_count);
}

async function postSettings(patch) {
  const res = await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch)
  });
  const data = await res.json();
  state.settings = data.settings;
  applySettingsUI();
}

function initSettingsModal() {
  const btn = document.getElementById("settingsBtn");
  const modal = document.getElementById("settingsModal");
  const backdrop = document.getElementById("modalBackdrop");
  const close = document.getElementById("closeSettings");
  function open() {
    modal.classList.remove("hidden");
    backdrop.classList.remove("hidden");
  }
  function hide() {
    modal.classList.add("hidden");
    backdrop.classList.add("hidden");
  }
  btn.addEventListener("click", open);
  close.addEventListener("click", hide);
  backdrop.addEventListener("click", hide);

  document.getElementById("nnueToggle").addEventListener("click", async () => {
    await postSettings({ nnue_enabled: !state.settings.nnue_enabled });
  });

  const slider = document.getElementById("speedSlider");
  slider.addEventListener("input", () => {
    document.getElementById("speedValue").innerText = Number(slider.value).toFixed(1);
  });
  slider.addEventListener("change", async () => {
    await postSettings({ speed_multiplier: Number(slider.value) });
  });

  const mapping = [200, 500, 1000, 3000];
  const vcSlider = document.getElementById("vehicleCountSlider");
  vcSlider.addEventListener("input", () => {
    const val = mapping[Number(vcSlider.value)];
    document.getElementById("vehicleCountValue").innerText = String(val);
  });
  vcSlider.addEventListener("change", async () => {
    const val = mapping[Number(vcSlider.value)];
    await postSettings({ target_vehicle_count: val });
  });
}

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "snapshot" || msg.type === "tick") {
      const p = msg.payload;
      // lights (state changes)
      for (const l of (p.lights || [])) upsertLight(l);

      // vehicles
      const ids = new Set();
      for (const v of (p.vehicles || [])) {
        ids.add(v.id);
        upsertVehicle(v);
      }
      pruneVehicles(ids);

      // Live count so you can verify settings vs reality
      const vc = document.getElementById("vehiclesOnMap");
      if (vc) vc.innerText = String(ids.size);

      // metrics
      const m = p.metrics || {};
      if (typeof m.avg_waiting_s === "number") {
        document.getElementById("avgWait").innerText = m.avg_waiting_s.toFixed(1);
      }
      // (removed) simulation-time since server start
      // push chart once per second
      if (typeof p.t === "number" && p.t - state.lastMetricPushT >= 1.0) {
        state.lastMetricPushT = p.t;
        pushMetric(p.t, (m.avg_waiting_s || 0), !!state.settings.nnue_enabled);
      }

      // refresh selection data + glow follow
      if (state.selection.type === "vehicle") {
        const data = state.lastVehiclePayload.get(state.selection.id);
        if (data) {
          const marker = state.vehicles.get(state.selection.id);
          if (marker) ensureGlowCircle(marker.getLatLng());
          renderSelectionDetails(data);
        }
      } else if (state.selection.type === "light") {
        const data = state.lastLightPayload.get(state.selection.id);
        if (data) {
          const marker = state.lights.get(state.selection.id);
          if (marker) ensureGlowCircle(marker.getLatLng());
          renderSelectionDetails(data);
        }
      }
    }
  };

  ws.onclose = () => {
    setTimeout(connectWS, 800);
  };
}

function animationLoop() {
  // Update marker positions smoothly using latest payload values.
  // Backend already sends 30Hz, but we still repaint via rAF.
  for (const [id, marker] of state.vehicles.entries()) {
    const v = state.lastVehiclePayload.get(id);
    if (!v) continue;
    marker.setLatLng([v.lat, v.lon]);
  }
  requestAnimationFrame(animationLoop);
}

async function main() {
  window.__CAR_SVG__ = await loadText("/static/assets/car.svg");
  window.__LIGHT_SVG__ = await loadText("/static/assets/traffic_light.svg");

  initMap();
  initChart();
  initSettingsModal();

  const boot = await (await fetch("/api/bootstrap")).json();
  drawRoads(boot.roads);
  for (const l of (boot.lights || [])) upsertLight(l);

  state.settings = boot.settings;
  applySettingsUI();

  // Populate legend icons so the colors match the markers.
  document.getElementById("legendCarMoving").src = svgDataUrl(window.__CAR_SVG__, "#38bdf8");
  document.getElementById("legendCarWaiting").src = svgDataUrl(window.__CAR_SVG__, "#fb7185");
  document.getElementById("legendCarArrived").src = svgDataUrl(window.__CAR_SVG__, "#a78bfa");
  document.getElementById("legendLightGreen").src = svgDataUrl(replaceFirst(replaceFirst(replaceFirst(window.__LIGHT_SVG__, 'fill="#ef4444"', 'fill="#334155"'), 'fill="#f59e0b"', 'fill="#334155"'), 'fill="#22c55e"', 'fill="#22c55e"'), "#ffffff");
  document.getElementById("legendLightYellow").src = svgDataUrl(replaceFirst(replaceFirst(replaceFirst(window.__LIGHT_SVG__, 'fill="#ef4444"', 'fill="#334155"'), 'fill="#f59e0b"', 'fill="#f59e0b"'), 'fill="#22c55e"', 'fill="#334155"'), "#ffffff");
  document.getElementById("legendLightRed").src = svgDataUrl(replaceFirst(replaceFirst(replaceFirst(window.__LIGHT_SVG__, 'fill="#ef4444"', 'fill="#ef4444"'), 'fill="#f59e0b"', 'fill="#334155"'), 'fill="#22c55e"', 'fill="#334155"'), "#ffffff");

  connectWS();
  requestAnimationFrame(animationLoop);
}

main().catch((e) => {
  console.error(e);
  alert("Failed to start UI. Check console.");
});
