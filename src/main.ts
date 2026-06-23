import "./style.css";
import { GPUEngine, type BodySnapshot, type CameraState, type CreationPreview, type Vec2 } from "./gpu-engine";
import { BodiesSidebar } from "./ui";
import { AudioManager } from "./audio";

type InteractionMode = "idle" | "create" | "pan" | "select";
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
let activeTool: "create" | "select" = "create";
let creationStyle: CreationStyle = "growing";
let selectedId: number | null = null;
let pointerId: number | null = null;
let spacePressed = false;
let holdStartedAt = 0;
let frozenRadius = BASE_RADIUS;
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
}
interface FadingHistory {
  points: HistoryPoint[];
  radius: number;
  alpha: number;
}
const bodyHistories = new Map<number, HistoryPoint[]>();
const fadingHistories: FadingHistory[] = [];

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
const spawnModeSelect = document.querySelector<HTMLSelectElement>("#creation-mode")!;
const fixedControlsDiv = document.querySelector<HTMLElement>("#fixed-controls")!;
const spawnMassSlider = document.querySelector<HTMLInputElement>("#spawn-mass")!;
const massPreview = document.querySelector<HTMLElement>("#mass-preview")!;

spawnModeSelect.addEventListener("change", () => {
  const isFixed = spawnModeSelect.value === "fixed";
  fixedControlsDiv.style.display = isFixed ? "flex" : "none";
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
});

btnToolSelect.addEventListener("click", () => {
  activeTool = "select";
  btnToolSelect.classList.add("is-active");
  btnToolCreate.classList.remove("is-active");
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
  const isFixed = spawnModeSelect.value === "fixed";
  const massVal = isFixed ? getSpawnMass() : (currentGrowthRadius() ** 2 * 0.15);
  const velocity = creationStyle === "vector"
    ? {
        x: (dragCurrentWorld.x - dragStartWorld.x) * VELOCITY_SCALE,
        y: (dragCurrentWorld.y - dragStartWorld.y) * VELOCITY_SCALE,
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
  overlayCtx.lineWidth = 1.5;
  overlayCtx.setLineDash([4, 4]);
  overlayCtx.lineDashOffset = -((performance.now() / 150) % 8);
  overlayCtx.shadowColor = "rgba(255, 255, 255, 0.3)";
  overlayCtx.shadowBlur = 4;

  if (points.length > 0) {
    overlayCtx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      overlayCtx.lineTo(points[i].x, points[i].y);
    }
  }
  overlayCtx.stroke();
  overlayCtx.setLineDash([]);
  overlayCtx.shadowBlur = 0;

  // Draw text info bubble
  const startScreen = worldToScreen(dragStartWorld);
  const speedVal = Math.hypot(velocity.x, velocity.y);

  overlayCtx.fillStyle = "rgba(19, 20, 22, 0.9)";
  overlayCtx.strokeStyle = "rgba(255, 255, 255, 0.12)";
  overlayCtx.lineWidth = 1.5;

  const textLines = [
    `Масса: ${Math.round(massVal).toLocaleString("ru-RU")}`,
    `Скорость: ${Math.round(speedVal).toLocaleString("ru-RU")} ед/с`,
  ];

  const bubbleWidth = 140;
  const bubbleHeight = 46;
  const bubbleX = startScreen.x + 15;
  const bubbleY = startScreen.y - 50;

  overlayCtx.beginPath();
  overlayCtx.roundRect(bubbleX, bubbleY, bubbleWidth, bubbleHeight, 8);
  overlayCtx.fill();
  overlayCtx.stroke();

  overlayCtx.fillStyle = "#f3f4f6";
  overlayCtx.font = "11px Inter, \"Segoe UI\", system-ui, sans-serif";
  overlayCtx.fillText(textLines[0], bubbleX + 12, bubbleY + 18);
  overlayCtx.fillText(textLines[1], bubbleX + 12, bubbleY + 32);
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
  if (isPaused) {
    lastUiUpdate = 0;
  }
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
  return {
    position: dragStartWorld,
    radius: currentGrowthRadius(now),
    vectorEnd: creationStyle === "vector" ? dragCurrentWorld : undefined,
  };
}

function endInteraction(): void {
  interactionMode = "idle";
  creationStyle = "growing";
  pointerId = null;
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
  } else if (interactionMode === "create") {
    dragCurrentWorld = screenToWorld(pointerScreenPosition(event));
    const velocity = creationStyle === "vector"
      ? {
          x: (dragCurrentWorld.x - dragStartWorld.x) * VELOCITY_SCALE,
          y: (dragCurrentWorld.y - dragStartWorld.y) * VELOCITY_SCALE,
        }
      : { x: 0, y: 0 };
    const radius = currentGrowthRadius();
    const isFixed = spawnModeSelect.value === "fixed";
    const mass = isFixed ? getSpawnMass() : undefined;
    engine.injectBody(dragStartWorld, velocity, radius, mass);
    if (isPaused) {
      lastUiUpdate = 0;
    }

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
    renderPoints.push({
      pos: boundaryPoint,
      vel: bodyVel,
      mass: bodyMass
    });
  }

  if (renderPoints.length < 2) return;

  overlayCtx.lineCap = "butt";
  overlayCtx.lineJoin = "round";
  overlayCtx.lineWidth = Math.max(1.0, Math.min(2.5, 0.6 + Math.sqrt(bodyMass) * 0.04));

  const len = renderPoints.length;
  for (let i = 0; i < len - 1; i++) {
    const pA = renderPoints[i];
    const pB = renderPoints[i + 1];

    const t = i / (len - 1);
    const segmentAlpha = t * baseAlpha * overallAlpha;

    const colorStr = getHeatColor(pB.mass, pB.vel.x, pB.vel.y, segmentAlpha);

    overlayCtx.beginPath();
    const sA = worldToScreen(pA.pos);
    const sB = worldToScreen(pB.pos);

    overlayCtx.moveTo(sA.x, sA.y);

    const d = Math.hypot(pB.pos.x - pA.pos.x, pB.pos.y - pA.pos.y);
    const sSpeedA = Math.hypot(pA.vel.x, pA.vel.y);
    const sSpeedB = Math.hypot(pB.vel.x, pB.vel.y);

    const kA = sSpeedA > 0.0001 ? d / (3 * sSpeedA) : 0;
    const kB = sSpeedB > 0.0001 ? d / (3 * sSpeedB) : 0;

    const cp1 = worldToScreen({
      x: pA.pos.x + pA.vel.x * kA,
      y: pA.pos.y + pA.vel.y * kA
    });
    const cp2 = worldToScreen({
      x: pB.pos.x - pB.vel.x * kB,
      y: pB.pos.y - pB.vel.y * kB
    });

    overlayCtx.bezierCurveTo(cp1.x, cp1.y, cp2.x, cp2.y, sB.x, sB.y);

    overlayCtx.strokeStyle = colorStr;
    overlayCtx.stroke();
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
      engine.step(FIXED_STEP);
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

        // Clean up histories for bodies that no longer exist, moving them to fadingHistories
        for (const [id, history] of bodyHistories.entries()) {
          if (!freshIds.has(id)) {
            const oldBody = snapshots.find((b) => b.id === id);
            const radius = oldBody ? oldBody.radius : 8;
            if (history.length >= 2) {
              fadingHistories.push({
                points: history,
                radius,
                alpha: 1.0
              });
            }
            bodyHistories.delete(id);
          }
        }

        // Add current positions to histories
        for (const body of fresh) {
          let history = bodyHistories.get(body.id);
          if (!history) {
            history = [];
            bodyHistories.set(body.id, history);
          }
          history.push({
            pos: { x: body.position.x, y: body.position.y },
            vel: { x: body.velocity.x, y: body.velocity.y },
            mass: body.mass
          });

          // Scale maximum history length based on body mass (large bodies get longer trails)
          let maxLen = 15;
          if (body.mass > 1000) maxLen = 45;
          else if (body.mass > 100) maxLen = 30;
          else if (body.mass < 15) maxLen = 6;

          // Multiply max length by timescale factor to preserve the trail's physical duration at high speeds
          const scaleFactor = Math.max(1.0, timescale);
          const scaledMaxLen = Math.round(maxLen * scaleFactor);

          while (history.length > scaledMaxLen) {
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
                const baseShake = Math.sqrt(oldB.mass) * 1.5;
                const zoomFactor = Math.min(1.0, camera.zoom);
                const shakeAdded = baseShake * factor * zoomFactor;
                shakeIntensity = Math.min(40, shakeIntensity + shakeAdded);
                audioManager.playCollision(oldB.mass, factor, camera.zoom);
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

  // Camera Lock logic
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

  // Draw smooth, tapered, and fading historical trails for active bodies on the overlay canvas
  // Apply a 1-frame lead projection to compensate for asynchronous WebGPU compositing latency
  const leadTime = 1.0 * FIXED_STEP * timescale;
  const dt = simTime - snapshotSimTime + leadTime;

  // 1. Render active trails
  for (const body of snapshots) {
    const history = bodyHistories.get(body.id);
    if (!history || history.length < 2) continue;

    // Calculate gravitational acceleration on the CPU to dead-reckon curved orbits accurately
    let ax = 0;
    let ay = 0;
    const G = 9500.0;
    const SOFTENING = 12.0;
    for (const att of snapshots) {
      if (att.id === body.id || att.mass <= 0) continue;
      const dx = att.position.x - body.position.x;
      const dy = att.position.y - body.position.y;
      const distSq = dx * dx + dy * dy + SOFTENING * SOFTENING;
      const dist = Math.sqrt(distSq);
      const pull = (G * att.mass) / (distSq * dist);
      ax += dx * pull;
      ay += dy * pull;
    }

    // Quadratic dead reckoning: position + velocity * dt + 0.5 * acceleration * dt^2
    const estX = body.position.x + body.velocity.x * dt + 0.5 * ax * dt * dt;
    const estY = body.position.y + body.velocity.y * dt + 0.5 * ay * dt * dt;
    const estVel = {
      x: body.velocity.x + ax * dt,
      y: body.velocity.y + ay * dt
    };

    const baseAlpha = body.mass > 1000 ? 0.32 : 0.20;

    drawSplineTrail(history, estX, estY, body.radius, body.mass, estVel, baseAlpha, 1.0);
  }

  // 2. Render and update fading trails (vanished bodies)
  for (let i = fadingHistories.length - 1; i >= 0; i--) {
    const fade = fadingHistories[i];
    
    // Decrement fade opacity
    fade.alpha -= 0.015; // slow dissolution over ~66 frames
    if (fade.alpha <= 0) {
      fadingHistories.splice(i, 1);
      continue;
    }

    const points = fade.points;
    const lastP = points[points.length - 1];
    
    // Project the fading trail slightly using its last velocity to keep it moving as it dies
    const estX = lastP.pos.x + lastP.vel.x * dt;
    const estY = lastP.pos.y + lastP.vel.y * dt;
    
    const baseAlpha = lastP.mass > 1000 ? 0.32 : 0.20;
    
    drawSplineTrail(points, estX, estY, fade.radius, lastP.mass, lastP.vel, baseAlpha, fade.alpha);
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

  engine.render(renderCamera, creationPreview(now));
  requestAnimationFrame(frame);
}

window.addEventListener("resize", resize);
resize();
sidebar.update(snapshots);
requestAnimationFrame(frame);

window.addEventListener("click", () => audioManager.resume(), { once: true });
window.addEventListener("pointerdown", () => audioManager.resume(), { once: true });
