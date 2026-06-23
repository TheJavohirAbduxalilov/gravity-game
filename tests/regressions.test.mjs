import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const engine = readFileSync(new URL("../src/gpu-engine.ts", import.meta.url), "utf8");
const main = readFileSync(new URL("../src/main.ts", import.meta.url), "utf8");
const ui = readFileSync(new URL("../src/ui.ts", import.meta.url), "utf8");
const styles = readFileSync(new URL("../src/style.css", import.meta.url), "utf8");
const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");

test("all 4096 slots contain the same body type", () => {
  assert.match(engine, /export const BODY_COUNT = 4096/);
  assert.doesNotMatch(engine, /MAIN_BODY_COUNT|isFragment/);
  assert.doesNotMatch(ui, /isFragment|осколок/i);
});

test("gravity uses tiled all-pairs and kick-drift-kick", () => {
  assert.match(engine, /var<workgroup> bodyTile/);
  assert.match(engine, /tile \+= WORKGROUP_SIZE/);
  assert.match(engine, /currentBody\.velocity \+= 0\.5 \* acceleration \* params\.dt/);
  assert.match(engine, /currentBody\.position \+= currentBody\.velocity \* params\.dt/);
});

test("collisions use a spatial hash", () => {
  assert.match(engine, /fn buildSpatialHash/);
  assert.match(engine, /fn hashCell/);
  assert.match(engine, /atomicExchange\(&metadata\[HASH_HEAD_OFFSET \+ bucket\]/);
  assert.match(engine, /fn selectPartners/);
});

test("fragmentation atomically claims ordinary free slots", () => {
  assert.match(engine, /fn reserveGrowth/);
  assert.match(engine, /atomicCompareExchangeWeak\(&metadata\[SLOT_STATE_OFFSET \+ slot\], 0u, 1u\)/);
  assert.match(engine, /outputBodies\[claimed\] = Body/);
});

test("fragmentation responds to impact physics and conserves mass and momentum", () => {
  assert.match(engine, /let reducedMass = sourceBody\.mass \* other\.mass \/ totalMass/);
  assert.match(engine, /let impactImpulse = reducedMass \* normalSpeed/);
  assert.match(engine, /normalSpeed \* normalSpeed \+ 0\.35 \* tangentSpeed \* tangentSpeed/);
  assert.match(engine, /energy \/ totalMass < disruptionThreshold\(totalMass\)/);
  assert.match(engine, /fragmentCount\(energy, totalMass, reducedMass, impactImpulse, obliquity\)/);
  assert.match(engine, /let mass = MIN_FRAGMENT_MASS \+ distributableMass \* weight \/ weightSum/);
  assert.match(engine, /let pattern = fragmentPattern\(event, ordinal\) - meanPattern/);
  assert.match(engine, /let velocity = event\.centerVelocity \+ pattern \* speedScale/);
  assert.doesNotMatch(engine, /6\.28318530718 \* f32\(ordinal\) \/ count/);
  assert.doesNotMatch(engine, /0\.5 \* totalMass \* relativeSpeed \* relativeSpeed/);
});

test("fragment cascades are bounded by physical resolution and reaccretion", () => {
  assert.match(engine, /const MIN_FRAGMENT_MASS: f32 = 2\.0/);
  assert.match(engine, /const FRAGMENT_REARM_TIME: f32 = 1\.2/);
  assert.match(engine, /const MAX_FRAGMENT_GENERATION: f32 = 2\.0/);
  assert.match(engine, /let massLimitedCount = u32\(floor\(totalMass \/ MIN_FRAGMENT_MASS\)\)/);
  assert.match(engine, /if \(protectionActive \|\| generation >= MAX_FRAGMENT_GENERATION\) \{ return; \}/);
  assert.match(engine, /disruptive \|\| protectionActive \|\| generation >= MAX_FRAGMENT_GENERATION \|\| unresolvedCollision/);
  assert.match(engine, /params\.time \+ FRAGMENT_REARM_TIME, event\.generation \+ 1\.0/);
});

test("WGSL stays within baseline binding limits and avoids reserved identifiers", () => {
  assert.match(engine, /@group\(0\) @binding\(7\)/);
  assert.doesNotMatch(engine, /@group\(0\) @binding\((?:8|9|10|11)\)/);
  assert.doesNotMatch(engine, /\bself\b|\bactive\b/);
  assert.match(engine, /pushErrorScope\("validation"\)/);
});

test("grid is infinite, adaptive, uniformly spaced, and gravity-warped", () => {
  assert.match(engine, /private gridLod\(zoom: number\)/);
  assert.match(engine, /const desiredStep = 80 \/ Math\.max\(zoom/);
  assert.match(engine, /fineStep = decade \* 2[\s\S]*coarseStep = decade \* 5/);
  assert.match(engine, /return fineStep \* \(coarseStep \/ fineStep\) \*\* blend/);
  assert.match(engine, /appendGrid\(gridStep, 0\.18\)/);
  assert.doesNotMatch(engine, /appendGrid\((?:fineStep|coarseStep)/);
  assert.match(engine, /const marginX = warpMargin\(96 \* worldPerPixel\)/);
  assert.match(engine, /camera\.x - viewWidth \* 0\.5 - marginX/);
  assert.match(engine, /let towardBody = attractor\.position - world/);
  assert.match(engine, /world \+= direction \* distance \* compression/);
  assert.match(engine, /index >= u32\(uniforms\.state\.w\)/);
  assert.doesNotMatch(engine, /createGridVertices/);
});

test("grid deformation reads live GPU body positions every frame", () => {
  assert.match(engine, /gridAttractorIndices: array<u32>/);
  assert.match(engine, /let attractor = bodies\[gridAttractorIndices\[index\]\]/);
  assert.match(engine, /const data = new Uint32Array\(GRID_ATTRACTOR_COUNT\)/);
  assert.match(engine, /data\[index\] = body\.id/);
  assert.doesNotMatch(engine, /struct GridAttractor/);
  assert.doesNotMatch(engine, /data\[offset\] = body\.position\.x/);
});

test("grid compression is asymptotically unlimited without folding", () => {
  assert.match(engine, /let rawPull = depth \/ \(distance \+ softening\)/);
  assert.match(engine, /let compression = 1\.0 - exp\(-rawPull \/ max\(distance, 0\.0001\)\)/);
  assert.match(engine, /var world = input\.position/);
  assert.doesNotMatch(engine, /distance \* 0\.42|displacementLimit|limitedDisplacement/);
  assert.match(engine, /const warpMargin = \(base: number\)/);
  assert.match(engine, /4 \* this\.gridWarpDepth/);
  assert.match(engine, /GRID_VERTEX_CAPACITY = 524288/);
  assert.match(engine, /while \(this\.gridVertexCapacity < this\.gridVertexCount\)/);
});

test("body creation radius has no application-level maximum", () => {
  assert.doesNotMatch(main, /MAX_CREATION_RADIUS/);
  assert.match(main, /return BASE_RADIUS \+ Math\.max\(0, now - holdStartedAt\)/);
});

test("camera pans with the right mouse button and suppresses its menu", () => {
  assert.match(main, /event\.button === 2 \|\| \(event\.button === 0 && spacePressed\)/);
  assert.doesNotMatch(main, /event\.button === 1/);
  assert.match(main, /addEventListener\("contextmenu", \(event\) => event\.preventDefault\(\)\)/);
});

test("UI contains only the canvas and body sidebar", () => {
  assert.match(html, /<canvas id="space-canvas"><\/canvas>/);
  assert.match(html, /<aside class="sidebar"/);
  assert.doesNotMatch(html, /topbar|brand-mark|instruction|legend|camera-help|<footer/);
  assert.doesNotMatch(main, /mode-create|mode-select|notice/);
  assert.match(styles, /\.app-shell \{[\s\S]*?margin-right: 284px;[\s\S]*?padding: 0;/);
  assert.match(styles, /\.sidebar \{[\s\S]*?top: 0;[\s\S]*?right: 0;[\s\S]*?bottom: 0;/);
});

test("mobile layout does not force a desktop minimum width", () => {
  const mobile = styles.slice(styles.indexOf("@media (max-width: 650px)"));
  assert.match(mobile, /min-width:\s*0/);
  assert.doesNotMatch(mobile, /min-width:\s*620px/);
});
