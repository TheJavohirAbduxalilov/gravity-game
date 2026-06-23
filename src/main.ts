import "./style.css";
import { GPUEngine, type BodySnapshot, type CameraState, type CreationPreview, type Vec2 } from "./gpu-engine";
import { BodiesSidebar } from "./ui";
import { AudioManager } from "./audio";

type InteractionMode = "idle" | "create" | "pan" | "select" | "move-position";
type CreationMode = "dynamic" | "fixed" | "black-hole";
type CreationStyle = "growing" | "vector";

const canvas = document.querySelector<HTMLCanvasElement>("#space-canvas")!;

const engine = await GPUEngine.create(canvas);
const camera: CameraState = { x: 0, y: 0, zoom: 1 };
const customNames = new Map<number, string>();
const sidebar = new BodiesSidebar(focusBody, deleteBody, renameBody);
const audioManager = new AudioManager();

const FIXED_STEP = 1 / 90;
const MAX_FRAME_TIME = 0.05;
const MAX_SUBSTEPS = 5;
const BASE_RADIUS = 8;
const HOLD_GROWTH_RATE = 11;
const VECTOR_THRESHOLD_PX = 10;
const VELOCITY_SCALE = 0.7;
const MIN_ZOOM = 0.01;
const MAX_ZOOM = 50;

const BLACK_HOLE_RADIUS = 6;
const BLACK_HOLE_MASS = 5_000_000;

let width = 1;
let height = 1;
let lastTime = performance.now();
let accumulator = 0;
let lastUiUpdate = 0;
let frameCount = 0;
let fpsLastTime = performance.now();
let simTime = 0;
let snapshotSimTime = 0;
let snapshots: BodySnapshot[] = [];
let interactionMode: InteractionMode = "idle";
let activeTool: "create" | "select" | "move" = "create"; // let activeTool: "create" | "select"
let creationStyle: CreationStyle = "growing";
let selectedId: number | null = null;
let pointerId: number | null = null;
let spacePressed = false;
let holdStartedAt = 0;
let frozenRadius = BASE_RADIUS;
let draggedBodyId: number | null = null;
let dragSamples: { time: number; pos: Vec2 }[] = [];
let dragOffsetWorld: Vec2 = { x: 0, y: 0 };
let dragStartScreen: Vec2 = { x: 0, y: 0 };
let dragStartWorld: Vec2 = { x: 0, y: 0 };
let dragCurrentWorld: Vec2 = { x: 0, y: 0 };
let panLastScreen: Vec2 = { x: 0, y: 0 };
let snapshotPending = false;
let isPaused = false;
let timescale = 1.0;
let lockedBodyId: number | null = null;
let shakeIntensity = 0;
const deletedByIds = new Set<number>();
interface HistoryPoint {
  pos: Vec2;
  vel: Vec2;
  mass: number;
  dist: number;
}
const bodyHistories = new Map<number, HistoryPoint[]>();

const settings = {
  cameraShake: localStorage.getItem("setting_camera_shake") !== "false",
  collisionSounds: localStorage.getItem("setting_collision_sounds") !== "false",
  showTrails: localStorage.getItem("setting_show_trails") !== "false",
  showGrid: localStorage.getItem("setting_show_grid") !== "false",
  velocityInputMode: localStorage.getItem("setting_velocity_input_mode") || "opposite"
};

const btnOpenSettings = document.querySelector<HTMLButtonElement>("#btn-open-settings")!;
const settingsModal = document.querySelector<HTMLElement>("#settings-modal")!;
const btnCloseModal = document.querySelector<HTMLButtonElement>("#btn-close-modal")!;
const btnSaveSettings = document.querySelector<HTMLButtonElement>("#btn-save-settings")!;

const settingCameraShake = document.querySelector<HTMLInputElement>("#setting-camera-shake")!;
const settingCollisionSounds = document.querySelector<HTMLInputElement>("#setting-collision-sounds")!;
const settingShowTrails = document.querySelector<HTMLInputElement>("#setting-show-trails")!;
const settingShowGrid = document.querySelector<HTMLInputElement>("#setting-show-grid")!;
const settingVelocityMode = document.querySelector<HTMLSelectElement>("#setting-velocity-mode")!;

const pauseButton = document.querySelector<HTMLButtonElement>("#pause-button")!;
const pauseText = pauseButton.querySelector<HTMLElement>(".pause-text")!;
const btnToolCreate = document.querySelector<HTMLButtonElement>("#btn-tool-create")!;
const btnToolSelect = document.querySelector<HTMLButtonElement>("#btn-tool-select")!;
const timescaleDec = document.querySelector<HTMLButtonElement>("#timescale-dec")!;
const timescaleInc = document.querySelector<HTMLButtonElement>("#timescale-inc")!;
const timescaleValueEl = document.querySelector<HTMLElement>("#timescale-value")!;
const zoomDec = document.querySelector<HTMLButtonElement>("#zoom-dec")!;
const zoomInc = document.querySelector<HTMLButtonElement>("#zoom-inc")!;
const zoomValueEl = document.querySelector<HTMLElement>("#zoom-value")!;
const cameraCoordsEl = document.querySelector<HTMLElement>("#camera-coords")!;
const fpsValueEl = document.querySelector<HTMLElement>("#fps-value")!;
const bodiesCountEl = document.querySelector<HTMLElement>("#bodies-count")!;

const overlayCanvas = document.querySelector<HTMLCanvasElement>("#overlay-canvas")!;
const overlayCtx = overlayCanvas.getContext("2d")!;
const spawnModeSelect = document.querySelector<HTMLInputElement>("#creation-mode")!;
const fixedControlsDiv = document.querySelector<HTMLElement>("#fixed-controls")!;
const blackHoleControlsDiv = document.querySelector<HTMLElement>("#black-hole-controls")!;
const spawnMassSlider = document.querySelector<HTMLInputElement>("#spawn-mass")!;
const massPreview = document.querySelector<HTMLElement>("#mass-preview")!;
const btnToolMove = document.querySelector<HTMLButtonElement>("#btn-tool-move")!;

const segmentedButtons = document.querySelectorAll<HTMLButtonElement>("#creation-mode-segmented .segment-btn");
segmentedButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    segmentedButtons.forEach((b) => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    const val = btn.getAttribute("data-value")!;
    spawnModeSelect.value = val;
    spawnModeSelect.dispatchEvent(new Event("change"));
  });
});

spawnModeSelect.addEventListener("change", () => {
  const mode = spawnModeSelect.value as CreationMode;
  fixedControlsDiv.style.display = mode === "fixed" ? "flex" : "none";
  if (blackHoleControlsDiv) {
    if (mode === "black-hole") {
      blackHoleControlsDiv.removeAttribute("hidden");
    } else {
      blackHoleControlsDiv.setAttribute("hidden", "");
    }
  }
});

function getSpawnMass(): number {
  const v = parseFloat(spawnMassSlider.value);
  const minLog = Math.log(1);
  const maxLog = Math.log(500000);
  const scale = (maxLog - minLog) / 999;
  return Math.round(Math.exp(minLog + (v - 1) * scale));
}

massPreview.textContent = getSpawnMass().toLocaleString("ru-RU");

spawnMassSlider.addEventListener("input", () => {
  massPreview.textContent = getSpawnMass().toLocaleString("ru-RU");
});

const playIconPath = "M8 5v14l11-7z";
const pauseIconPath = "M6 19h4V5H6v14zm8-14v14h4V5h-4z";
const pauseSvgPath = pauseButton.querySelector("svg path")!;
pauseSvgPath.setAttribute("d", pauseIconPath);

pauseButton.addEventListener("click", () => {
  isPaused = !isPaused;
  pauseButton.classList.toggle("is-active", isPaused);
  pauseText.textContent = isPaused ? "Продолжить" : "Пауза";
  pauseSvgPath.setAttribute("d", isPaused ? playIconPath : pauseIconPath);
});

btnToolCreate.addEventListener("click", () => {
  activeTool = "create";
  btnToolCreate.classList.add("is-active");
  btnToolSelect.classList.remove("is-active");
  btnToolMove.classList.remove("is-active");
});

btnToolSelect.addEventListener("click", () => {
  activeTool = "select";
  btnToolSelect.classList.add("is-active");
  btnToolCreate.classList.remove("is-active");
  btnToolMove.classList.remove("is-active");
});

btnToolMove.addEventListener("click", () => {
  activeTool = "move";
  btnToolMove.classList.add("is-active");
  btnToolCreate.classList.remove("is-active");
  btnToolSelect.classList.remove("is-active");
});

timescaleDec.addEventListener("click", () => {
  if (timescale <= 1.05) {
    timescale = Math.max(0.1, timescale - 0.1);
  } else {
    timescale = Math.max(1.0, timescale - 0.5);
  }
  timescaleValueEl.textContent = `${timescale.toFixed(1)}x`;
});

timescaleInc.addEventListener("click", () => {
  if (timescale < 0.95) {
    timescale = Math.min(1.0, timescale + 0.1);
  } else {
    timescale = Math.min(5.0, timescale + 0.5);
  }
  timescaleValueEl.textContent = `${timescale.toFixed(1)}x`;
});

zoomDec.addEventListener("click", () => {
  camera.zoom = Math.max(MIN_ZOOM, camera.zoom / 1.3);
  updateToolbarReadouts();
});

zoomInc.addEventListener("click", () => {
  camera.zoom = Math.min(MAX_ZOOM, camera.zoom * 1.3);
  updateToolbarReadouts();
});

btnOpenSettings.addEventListener("click", () => {
  settingCameraShake.checked = settings.cameraShake;
  settingCollisionSounds.checked = settings.collisionSounds;
  settingShowTrails.checked = settings.showTrails;
  settingShowGrid.checked = settings.showGrid;
  settingVelocityMode.value = settings.velocityInputMode;
  
  settingsModal.classList.add("is-open");
  settingsModal.setAttribute("aria-hidden", "false");
});

const closeModal = () => {
  settingsModal.classList.remove("is-open");
  settingsModal.setAttribute("aria-hidden", "true");
};

btnCloseModal.addEventListener("click", closeModal);

btnSaveSettings.addEventListener("click", () => {
  settings.cameraShake = settingCameraShake.checked;
  settings.collisionSounds = settingCollisionSounds.checked;
  settings.showTrails = settingShowTrails.checked;
  settings.showGrid = settingShowGrid.checked;
  settings.velocityInputMode = settingVelocityMode.value;
  
  localStorage.setItem("setting_camera_shake", String(settings.cameraShake));
  localStorage.setItem("setting_collision_sounds", String(settings.collisionSounds));
  localStorage.setItem("setting_show_trails", String(settings.showTrails));
  localStorage.setItem("setting_show_grid", String(settings.showGrid));
  localStorage.setItem("setting_velocity_input_mode", settings.velocityInputMode);
  
  closeModal();
});

settingsModal.addEventListener("click", (event) => {
  if (event.target === settingsModal) {
    closeModal();
  }
});

function updateToolbarReadouts(): void {
  zoomValueEl.textContent = `${camera.zoom.toFixed(2)}x`;
  cameraCoordsEl.textContent = `${Math.round(camera.x)}, ${Math.round(camera.y)}`;
}

function worldToScreen(point: Vec2): Vec2 {
  return {
    x: (point.x - camera.x) * camera.zoom + width / 2,
    y: (point.y - camera.y) * camera.zoom + height / 2,
  };
}

function drawOrbitPrediction(): void {
  const massVal = creationInjectionMass() ?? (creationInjectionRadius() ** 2 * 0.15);
  const mult = settings.velocityInputMode === "opposite" ? -1 : 1;
  const velocity = creationStyle === "vector"
    ? {
        x: (dragCurrentWorld.x - dragStartWorld.x) * VELOCITY_SCALE * mult,
        y: (dragCurrentWorld.y - dragStartWorld.y) * VELOCITY_SCALE * mult,
      }
    : { x: 0, y: 0 };

  const attractors = snapshots.filter((b) => b.mass > 0);
  const G = 9500.0;
  const SOFTENING = 12.0;
  const steps = 350;
  const dt = 0.03;

  let p = { x: dragStartWorld.x, y: dragStartWorld.y };
  let v = { x: velocity.x, y: velocity.y };
  const points: Vec2[] = [];
  points.push(worldToScreen(p));

  for (let i = 0; i < steps; i++) {
    let ax = 0;
    let ay = 0;
    for (const att of attractors) {
      const dx = att.position.x - p.x;
      const dy = att.position.y - p.y;
      const distSq = dx * dx + dy * dy + SOFTENING * SOFTENING;
      const dist = Math.sqrt(distSq);
      const pull = (G * att.mass) / (distSq * dist);
      ax += dx * pull;
      ay += dy * pull;
    }
    v.x += ax * dt;
    v.y += ay * dt;
    p.x += v.x * dt;
    p.y += v.y * dt;
    points.push(worldToScreen(p));
  }

  overlayCtx.beginPath();
  overlayCtx.strokeStyle = "rgba(255, 255, 255, 0.75)";
  overlayCtx.lineWidth = 1;
  overlayCtx.setLineDash([4, 4]);
  overlayCtx.lineDashOffset = -((performance.now() / 150) % 8);

  if (points.length > 0) {
    overlayCtx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      overlayCtx.lineTo(points[i].x, points[i].y);
    }
  }
  overlayCtx.stroke();
  overlayCtx.setLineDash([]);

  // Draw text info bubble
  const startScreen = worldToScreen(dragStartWorld);
  const speedVal = Math.hypot(velocity.x, velocity.y);

  overlayCtx.fillStyle = "rgba(13, 15, 19, 0.94)";
  overlayCtx.strokeStyle = "rgba(91, 99, 113, 0.9)";
  overlayCtx.lineWidth = 1;

  const textLines = [
    `Масса: ${Math.round(massVal).toLocaleString("ru-RU")}`,
    `Скорость: ${Math.round(speedVal).toLocaleString("ru-RU")} ед/с`,
  ];

  const bubbleWidth = 136;
  const bubbleHeight = 42;
  const bubbleX = startScreen.x + 15;
  const bubbleY = startScreen.y - 46;

  overlayCtx.beginPath();
  overlayCtx.roundRect(bubbleX, bubbleY, bubbleWidth, bubbleHeight, 2);
  overlayCtx.fill();
  overlayCtx.stroke();

  overlayCtx.fillStyle = "#e6e8ec";
  overlayCtx.font = "11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  overlayCtx.fillText(textLines[0], bubbleX + 8, bubbleY + 16);
  overlayCtx.fillStyle = "#a2a9b5";
  overlayCtx.fillText(textLines[1], bubbleX + 8, bubbleY + 30);
}

function resize(): void {
  const rect = canvas.getBoundingClientRect();
  width = Math.max(1, rect.width);
  height = Math.max(1, rect.height);
  engine.resize(width, height);

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  overlayCanvas.width = width * dpr;
  overlayCanvas.height = height * dpr;
  overlayCtx.resetTransform();
  overlayCtx.scale(dpr, dpr);
}

function pointerScreenPosition(event: PointerEvent | WheelEvent): Vec2 {
  const rect = canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function screenToWorld(point: Vec2): Vec2 {
  return {
    x: camera.x + (point.x - width / 2) / camera.zoom,
    y: camera.y + (point.y - height / 2) / camera.zoom,
  };
}

function currentGrowthRadius(now = performance.now()): number {
  if (spawnModeSelect.value === "fixed") {
    const mass = getSpawnMass();
    return Math.sqrt(mass / 0.15);
  }
  if (creationStyle === "vector") return frozenRadius;
  return BASE_RADIUS + Math.max(0, now - holdStartedAt) / 1000 * HOLD_GROWTH_RATE;
}

function creationInjectionRadius(now?: number): number {
  const mode = spawnModeSelect.value as CreationMode;
  if (mode === "black-hole") return BLACK_HOLE_RADIUS;
  return currentGrowthRadius(now);
}

function creationInjectionMass(): number | undefined {
  const mode = spawnModeSelect.value as CreationMode;
  if (mode === "black-hole") return BLACK_HOLE_MASS;
  const isFixed = mode === "fixed";
  return isFixed ? getSpawnMass() : undefined;
}

function focusBody(id: number): void {
  lockedBodyId = id;
  selectBody(id);
}

function selectBody(id: number | null): void {
  selectedId = id;
  if (id === null) {
    lockedBodyId = null;
  }
  engine.setSelected(id ?? -1);
  sidebar.setSelected(id);
  sidebar.update(snapshots);
}

function deleteBody(id: number): void {
  deletedByIds.add(id);
  if (selectedId === id) selectBody(null);
  engine.deleteBody(id);
  lastUiUpdate = 0;
}

function renameBody(id: number, name: string): void {
  customNames.set(id, name);
  const body = snapshots.find((b) => b.id === id);
  if (body) {
    body.name = name;
  }
  sidebar.update(snapshots);
}


function creationPreview(now: number): CreationPreview | null {
  if (interactionMode !== "create") return null;
  const mult = settings.velocityInputMode === "opposite" ? -1 : 1;
  const vectorEnd = creationStyle === "vector"
    ? {
        x: dragStartWorld.x + (dragCurrentWorld.x - dragStartWorld.x) * mult,
        y: dragStartWorld.y + (dragCurrentWorld.y - dragStartWorld.y) * mult,
      }
    : undefined;
  return {
    position: dragStartWorld,
    radius: creationInjectionRadius(now),
    vectorEnd,
  };
}

function endInteraction(): void {
  interactionMode = "idle";
  creationStyle = "growing";
  pointerId = null;
  draggedBodyId = null;
  dragSamples = [];
  canvas.classList.remove("is-aiming", "is-panning");
  canvas.classList.toggle("space-ready", spacePressed);
  audioManager.stopCreation(0);
}

canvas.addEventListener("pointerdown", (event) => {
  audioManager.resume();
  const shouldPan = event.button === 2 || (event.button === 0 && spacePressed);
  if (!shouldPan && event.button !== 0) return;
  event.preventDefault();

  pointerId = event.pointerId;
  canvas.setPointerCapture(event.pointerId);

  if (shouldPan) {
    interactionMode = "pan";
    panLastScreen = pointerScreenPosition(event);
    canvas.classList.add("is-panning");
    canvas.classList.remove("space-ready");
    return;
  }

  if (activeTool === "select") {
    interactionMode = "select";
    dragStartScreen = pointerScreenPosition(event);
    dragStartWorld = screenToWorld(dragStartScreen);
    return;
  }

  if (activeTool === "move") {
    const screenPosition = pointerScreenPosition(event);
    const startWorld = screenToWorld(screenPosition);
    let closestBody: typeof snapshots[0] | null = null;
    let minDistance = Infinity;
    const clickThreshold = 18 / camera.zoom; // 18 pixels in world space

    for (const body of snapshots) {
      const dist = Math.hypot(body.position.x - startWorld.x, body.position.y - startWorld.y);
      const hitRadius = Math.max(body.radius, clickThreshold);
      if (dist <= hitRadius && dist < minDistance) {
        minDistance = dist;
        closestBody = body;
      }
    }

    if (closestBody) {
      interactionMode = "move-position";
      draggedBodyId = closestBody.id;
      dragOffsetWorld = {
        x: closestBody.position.x - startWorld.x,
        y: closestBody.position.y - startWorld.y,
      };
      dragCurrentWorld = { ...closestBody.position };
      dragSamples = [{ time: performance.now(), pos: { ...closestBody.position } }];
      engine.updateBody(draggedBodyId, closestBody.position, { x: 0, y: 0 });
      lastUiUpdate = 0;
    } else {
      interactionMode = "idle";
    }
    return;
  }

  interactionMode = "create";
  creationStyle = "growing";
  holdStartedAt = performance.now();
  frozenRadius = BASE_RADIUS;
  dragStartScreen = pointerScreenPosition(event);
  dragStartWorld = screenToWorld(dragStartScreen);
  dragCurrentWorld = { ...dragStartWorld };
  canvas.classList.add("is-aiming");
  audioManager.startCreation();
});

canvas.addEventListener("pointermove", (event) => {
  if (event.pointerId !== pointerId) return;
  const screenPosition = pointerScreenPosition(event);

  if (interactionMode === "pan") {
    camera.x -= (screenPosition.x - panLastScreen.x) / camera.zoom;
    camera.y -= (screenPosition.y - panLastScreen.y) / camera.zoom;
    panLastScreen = screenPosition;
    lockedBodyId = null;
    return;
  }

  if (interactionMode === "select") {
    return;
  }

  if (interactionMode === "move-position" && draggedBodyId !== null) {
    const startWorld = screenToWorld(screenPosition);
    const newPos = {
      x: startWorld.x + dragOffsetWorld.x,
      y: startWorld.y + dragOffsetWorld.y,
    };
    engine.updateBody(draggedBodyId, newPos, { x: 0, y: 0 });
    dragSamples.push({ time: performance.now(), pos: newPos });
    const now = performance.now();
    while (dragSamples.length > 2 && now - dragSamples[0].time > 150) {
      dragSamples.shift();
    }
    dragCurrentWorld = newPos;
    return;
  }

  if (interactionMode === "create") {
    dragCurrentWorld = screenToWorld(screenPosition);
    if (
      creationStyle === "growing" &&
      Math.hypot(screenPosition.x - dragStartScreen.x, screenPosition.y - dragStartScreen.y) > VECTOR_THRESHOLD_PX
    ) {
      frozenRadius = currentGrowthRadius();
      creationStyle = "vector";
    }
    audioManager.updateCreationPitch(currentGrowthRadius());
  }
});

canvas.addEventListener("pointerup", (event) => {
  if (event.pointerId !== pointerId) return;
  
  if (interactionMode === "select") {
    const screenPosition = pointerScreenPosition(event);
    const distPx = Math.hypot(screenPosition.x - dragStartScreen.x, screenPosition.y - dragStartScreen.y);
    if (distPx < 6) {
      // It's a click! Let's select the body under the cursor
      let closestBody: typeof snapshots[0] | null = null;
      let minDistance = Infinity;
      const clickThreshold = 18 / camera.zoom; // 18 pixels in world space

      for (const body of snapshots) {
        const dist = Math.hypot(body.position.x - dragStartWorld.x, body.position.y - dragStartWorld.y);
        const hitRadius = Math.max(body.radius, clickThreshold);
        if (dist <= hitRadius && dist < minDistance) {
          minDistance = dist;
          closestBody = body;
        }
      }

      if (closestBody) {
        selectBody(closestBody.id);
      } else {
        selectBody(null);
      }
    }
  } else if (interactionMode === "move-position" && draggedBodyId !== null) {
    const now = performance.now();
    while (dragSamples.length > 2 && now - dragSamples[0].time > 100) {
      dragSamples.shift();
    }

    let throwVelocity = { x: 0, y: 0 };
    if (dragSamples.length >= 2) {
      const first = dragSamples[0];
      const last = dragSamples[dragSamples.length - 1];
      const dt = (last.time - first.time) / 1000;
      if (dt > 0.001) {
        const THROW_VELOCITY_SCALE = 0.4;
        throwVelocity = {
          x: ((last.pos.x - first.pos.x) / dt) * THROW_VELOCITY_SCALE,
          y: ((last.pos.y - first.pos.y) / dt) * THROW_VELOCITY_SCALE,
        };
      }
    }

    const finalPos = dragSamples[dragSamples.length - 1]?.pos ?? dragStartWorld;
    engine.updateBody(draggedBodyId, finalPos, throwVelocity);
    
    draggedBodyId = null;
    dragSamples = [];
    lastUiUpdate = 0;
    endInteraction();
    return;
  } else if (interactionMode === "create") {
    dragCurrentWorld = screenToWorld(pointerScreenPosition(event));
    const mult = settings.velocityInputMode === "opposite" ? -1 : 1;
    const velocity = creationStyle === "vector"
      ? {
          x: (dragCurrentWorld.x - dragStartWorld.x) * VELOCITY_SCALE * mult,
          y: (dragCurrentWorld.y - dragStartWorld.y) * VELOCITY_SCALE * mult,
        }
      : { x: 0, y: 0 };
    const radius = creationInjectionRadius();
    const mass = creationInjectionMass();
    engine.injectBody(dragStartWorld, velocity, radius, mass);
    lastUiUpdate = 0;

    const speed = Math.hypot(velocity.x, velocity.y);
    audioManager.stopCreation(speed);
  }
  endInteraction();
});

canvas.addEventListener("pointercancel", endInteraction);
canvas.addEventListener("contextmenu", (event) => event.preventDefault());

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const cursor = pointerScreenPosition(event);
  const worldUnderCursor = screenToWorld(cursor);
  camera.zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, camera.zoom * Math.exp(-event.deltaY * 0.0015)));
  camera.x = worldUnderCursor.x - (cursor.x - width / 2) / camera.zoom;
  camera.y = worldUnderCursor.y - (cursor.y - height / 2) / camera.zoom;
}, { passive: false });

window.addEventListener("keydown", (event) => {
  if (event.code !== "Space" || event.repeat) return;
  const target = event.target as HTMLElement | null;
  if (target?.matches("button, input, textarea, select")) return;
  event.preventDefault();
  spacePressed = true;
  if (interactionMode === "idle") canvas.classList.add("space-ready");
});

window.addEventListener("keyup", (event) => {
  if (event.code !== "Space") return;
  spacePressed = false;
  if (interactionMode === "idle") canvas.classList.remove("space-ready");
});

window.addEventListener("blur", () => {
  spacePressed = false;
  if (interactionMode === "idle") canvas.classList.remove("space-ready");
});

function getHeatColor(mass: number, vx: number, vy: number, alpha: number): string {
  const speed = Math.hypot(vx, vy);
  const t = 1.0 - Math.exp(-(mass * 0.012 + speed * 0.005));
  let r = 0, g = 0, b = 0;
  if (t < 0.35) {
    const k = t / 0.35;
    r = 100 + (180 - 100) * k;
    g = 105 + (30 - 105) * k;
    b = 115 + (20 - 115) * k;
  } else if (t < 0.7) {
    const k = (t - 0.35) / 0.35;
    r = 180 + (255 - 180) * k;
    g = 30 + (115 - 30) * k;
    b = 20 + (30 - 20) * k;
  } else {
    const k = (t - 0.7) / 0.3;
    r = 255;
    g = 115 + (215 - 115) * k;
    b = 30 + (100 - 30) * k;
  }
  return `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${alpha.toFixed(3)})`;
}

function drawSplineTrail(
  points: HistoryPoint[],
  estX: number,
  estY: number,
  bodyRadius: number,
  bodyMass: number,
  bodyVel: Vec2,
  baseAlpha: number,
  overallAlpha: number
): void {
  if (points.length < 2) return;

  const renderPoints: HistoryPoint[] = [];
  for (const p of points) {
    const dist = Math.hypot(p.pos.x - estX, p.pos.y - estY);
    if (dist >= bodyRadius) {
      renderPoints.push(p);
    }
  }

  const lastPoint = points[points.length - 1];
  const dx = lastPoint.pos.x - estX;
  const dy = lastPoint.pos.y - estY;
  const dist = Math.hypot(dx, dy);
  if (dist > bodyRadius && dist > 0.0001) {
    const boundaryPoint = {
      x: estX + (dx / dist) * bodyRadius,
      y: estY + (dy / dist) * bodyRadius
    };
    const boundaryDist = lastPoint.dist + (dist - bodyRadius);
    renderPoints.push({
      pos: boundaryPoint,
      vel: bodyVel,
      mass: bodyMass,
      dist: boundaryDist
    });
  }

  if (renderPoints.length < 2) return;

  // Spacing target on screen is 12 pixels. In world space, this is:
  const targetSpacingScreen = 12;
  const targetSpacingWorld = targetSpacingScreen / camera.zoom;

  const interpolated: Vec2[] = [];
  const minDist = renderPoints[0].dist;
  const maxDist = renderPoints[renderPoints.length - 1].dist;

  const startN = Math.ceil(minDist / targetSpacingWorld);
  const endN = Math.floor(maxDist / targetSpacingWorld);

  let segmentIdx = 0;
  for (let n = startN; n <= endN; n++) {
    const targetDist = n * targetSpacingWorld;
    // Find the segment containing targetDist
    while (segmentIdx < renderPoints.length - 1 && renderPoints[segmentIdx + 1].dist < targetDist) {
      segmentIdx++;
    }
    if (segmentIdx >= renderPoints.length - 1) break;

    const pA = renderPoints[segmentIdx];
    const pB = renderPoints[segmentIdx + 1];
    const denom = pB.dist - pA.dist;
    const t = denom > 0.0001 ? (targetDist - pA.dist) / denom : 0;

    interpolated.push({
      x: pA.pos.x + (pB.pos.x - pA.pos.x) * t,
      y: pA.pos.y + (pB.pos.y - pA.pos.y) * t
    });
  }

  // Convert to screen space
  const screenPoints = interpolated.map((p) => worldToScreen(p));

  // Always draw a point at the head (the boundary point) to prevent any gaps
  const lastRenderPoint = renderPoints[renderPoints.length - 1];
  const lastScreenPoint = worldToScreen(lastRenderPoint.pos);
  if (screenPoints.length > 0) {
    const lastInterp = screenPoints[screenPoints.length - 1];
    const distToHead = Math.hypot(lastScreenPoint.x - lastInterp.x, lastScreenPoint.y - lastInterp.y);
    if (distToHead > 3.0) {
      screenPoints.push(lastScreenPoint);
    }
  } else {
    screenPoints.push(lastScreenPoint);
  }

  const numInterp = screenPoints.length;
  // Calculate zoom factor for dot radius
  const zoomFactor = Math.pow(camera.zoom, 0.35);
  // Base dot size in pixels (independent of body mass)
  const baseDotSize = 1.3;

  for (let j = 0; j < numInterp; j++) {
    const sPos = screenPoints[j];

    // Performance optimization: skip off-screen dots
    if (sPos.x < -10 || sPos.x > width + 10 || sPos.y < -10 || sPos.y > height + 10) {
      continue;
    }



    // t goes from 0.0 (tail) to 1.0 (head)
    const t = j / Math.max(1, numInterp - 1);
    const alpha = t * baseAlpha * overallAlpha;

    // Scale dot radius: tapers to 40% at the tail, and adjusted by zoomFactor
    const size = baseDotSize * (0.4 + 0.6 * t) * zoomFactor;
    // Keep size bounded between 0.7px and 3.0px so it remains a crisp dot
    const dotRadius = Math.max(0.7, Math.min(3.0, size));

    overlayCtx.beginPath();
    overlayCtx.arc(sPos.x, sPos.y, dotRadius, 0, 2 * Math.PI);
    overlayCtx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
    overlayCtx.fill();
  }
}

function frame(now: number): void {
  frameCount++;
  if (now - fpsLastTime >= 1000) {
    const fps = Math.round((frameCount * 1000) / (now - fpsLastTime));
    fpsValueEl.textContent = `${fps}`;
    frameCount = 0;
    fpsLastTime = now;
  }

  const frameTime = Math.min((now - lastTime) / 1000, MAX_FRAME_TIME);
  lastTime = now;

  if (!isPaused) {
    accumulator += frameTime * timescale;
    let substeps = 0;
    const maxSteps = Math.ceil(20 * timescale);
    while (accumulator >= FIXED_STEP && substeps < maxSteps) {
      engine.step(FIXED_STEP, draggedBodyId !== null ? draggedBodyId : 16384);
      simTime += FIXED_STEP;
      accumulator -= FIXED_STEP;
      substeps += 1;
    }
    if (substeps === maxSteps) accumulator = 0;
  } else {
    accumulator = 0;
  }

  const uiInterval = Math.max(36, 180 / timescale);
  if (now - lastUiUpdate > uiInterval) {
    if (!snapshotPending) {
      snapshotPending = true;
      const snapTime = simTime;
      void engine.readSnapshot().then((fresh) => {
        const freshIds = new Set(fresh.map((b) => b.id));
        snapshotSimTime = snapTime;

        // Clean up histories for bodies that no longer exist
        for (const id of bodyHistories.keys()) {
          if (!freshIds.has(id)) {
            bodyHistories.delete(id);
          }
        }

        // Add current positions to histories
        for (const body of fresh) {
          let history = bodyHistories.get(body.id);
          let prevDist = 0;
          let prevPos = body.position;
          if (!history) {
            history = [];
            bodyHistories.set(body.id, history);
          } else if (history.length > 0) {
            const last = history[history.length - 1];
            prevDist = last.dist;
            prevPos = last.pos;
          }

          const stepDist = Math.hypot(body.position.x - prevPos.x, body.position.y - prevPos.y);
          const newDist = prevDist + stepDist;

          history.push({
            pos: { x: body.position.x, y: body.position.y },
            vel: { x: body.velocity.x, y: body.velocity.y },
            mass: body.mass,
            dist: newDist
          });

          // Scale maximum history length based on body mass (large bodies get longer trails)
          let maxLen = 40;
          if (body.mass > 1000) maxLen = 120;
          else if (body.mass > 100) maxLen = 80;
          else if (body.mass < 15) maxLen = 20;

          while (history.length > maxLen) {
            history.shift();
          }
        }

        // Detect collisions by checking disappeared bodies
        if (snapshots.length > 0) {
          for (const oldB of snapshots) {
            if (!freshIds.has(oldB.id)) {
              if (deletedByIds.has(oldB.id)) {
                deletedByIds.delete(oldB.id);
                continue;
              }
              const dist = Math.hypot(oldB.position.x - camera.x, oldB.position.y - camera.y);
              const viewWidth = width / camera.zoom;
              const viewHeight = height / camera.zoom;
              const viewRadius = Math.hypot(viewWidth, viewHeight) * 0.5;
              const maxDist = viewRadius * 3;
              const factor = Math.max(0, 1 - dist / maxDist);
              
              if (factor > 0) {
                if (settings.cameraShake) {
                  const baseShake = Math.sqrt(oldB.mass) * 1.5;
                  const zoomFactor = Math.min(1.0, camera.zoom);
                  const shakeAdded = baseShake * factor * zoomFactor;
                  shakeIntensity = Math.min(40, shakeIntensity + shakeAdded);
                }
                if (settings.collisionSounds) {
                  audioManager.playCollision(oldB.mass, factor, camera.zoom);
                }
              }
            }
          }
        }

        // Clean up custom names for bodies that no longer exist
        for (const id of customNames.keys()) {
          if (!freshIds.has(id)) {
            customNames.delete(id);
          }
        }

        // Map custom names to fresh snapshots
        for (const body of fresh) {
          const custom = customNames.get(body.id);
          if (custom !== undefined) {
            body.name = custom;
          }
        }

        snapshots = fresh;
        sidebar.update(snapshots);
        bodiesCountEl.textContent = `${snapshots.length}`;
      }).finally(() => {
        snapshotPending = false;
      });
    }
    lastUiUpdate = now;
  }

  // 1. Calculate predictedSnapshots using CPU dead reckoning
  // If a body is being dragged, update its position and zero velocity directly in the snapshots array
  if (draggedBodyId !== null) {
    const dragged = snapshots.find((b) => b.id === draggedBodyId);
    if (dragged) {
      dragged.position = { ...dragCurrentWorld };
      dragged.velocity = { x: 0, y: 0 };
    }
  }

  // Camera Lock logic using snapshots
  if (lockedBodyId !== null) {
    const locked = snapshots.find((b) => b.id === lockedBodyId);
    if (locked) {
      const lerpFactor = 0.08;
      camera.x += (locked.position.x - camera.x) * lerpFactor;
      camera.y += (locked.position.y - camera.y) * lerpFactor;
    } else {
      lockedBodyId = null;
    }
  }
  
  overlayCtx.clearRect(0, 0, width, height);

  const leadTime = 0.0 * FIXED_STEP * timescale;
  const dt = simTime - snapshotSimTime + leadTime;

  // 2. Render active trails using snapshots and O(1) linear prediction
  if (settings.showTrails) {
    for (const body of snapshots) {
      const history = bodyHistories.get(body.id);
      if (!history || history.length < 2) continue;

      const estX = body.position.x + body.velocity.x * dt;
      const estY = body.position.y + body.velocity.y * dt;
      const estVel = body.velocity;

      const baseAlpha = body.mass > 1000 ? 0.70 : 0.45;
      drawSplineTrail(history, estX, estY, body.radius, body.mass, estVel, baseAlpha, 1.0);
    }
  }

  if (interactionMode === "create") {
    drawOrbitPrediction();
  }

  updateToolbarReadouts();

  // Render with camera shake applied to rendering view coordinates
  let rx = camera.x;
  let ry = camera.y;
  if (shakeIntensity > 0.05) {
    rx += (Math.random() - 0.5) * 2 * shakeIntensity / camera.zoom;
    ry += (Math.random() - 0.5) * 2 * shakeIntensity / camera.zoom;
    shakeIntensity *= 0.88;
  }
  const renderCamera = { x: rx, y: ry, zoom: camera.zoom };

  engine.render(renderCamera, creationPreview(now), settings.showGrid, snapshots);
  requestAnimationFrame(frame);
}

window.addEventListener("resize", resize);
resize();
sidebar.update(snapshots);
requestAnimationFrame(frame);

window.addEventListener("click", () => audioManager.resume(), { once: true });
window.addEventListener("pointerdown", () => audioManager.resume(), { once: true });
