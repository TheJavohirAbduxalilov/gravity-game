import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const engine = readFileSync(new URL("../src/gpu-engine.ts", import.meta.url), "utf8");
const main = readFileSync(new URL("../src/main.ts", import.meta.url), "utf8");
const ui = readFileSync(new URL("../src/ui.ts", import.meta.url), "utf8");
const styles = readFileSync(new URL("../src/style.css", import.meta.url), "utf8");
const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");

test("all 16384 slots contain the same body type", () => {
  assert.match(engine, /export const BODY_COUNT = 16384/);
  assert.doesNotMatch(engine, /MAIN_BODY_COUNT|isFragment/);
  assert.doesNotMatch(ui, /isFragment|осколок/i);
});

test("gravity uses a Morton-sorted GPU linear quadtree for every active body", () => {
  assert.match(engine, /const TREE_DEPTH = 6/);
  assert.match(engine, /fn mortonCode\(position: vec2<f32>\) -> u32/);
  assert.match(engine, /atomicAdd\(&counters\.treeBucketCounts\[mortonCode\(body\.position\)\], 1u\)/);
  assert.match(engine, /fn prefixTreeBuckets\(\)/);
  assert.match(engine, /counters\.treeSortedIndices\[sortedSlot\] = bodyIndex/);
  assert.match(engine, /fn buildTreeHierarchy/);
  assert.match(engine, /let childBase = childOffset \+ local \* 4u/);
  assert.match(engine, /fn treeAcceleration\(sourceBody: Body, sourceIndex: u32, useDrift: bool\)/);
  assert.match(engine, /node\.size \/ max\(distance, 0\.0001\) < TREE_THETA/);
  assert.doesNotMatch(engine, /ATTRACTOR_THRESHOLD|MAX_ATTRACTORS|collectAttractors/);
});

test("kick-drift-kick rebuilds the gravity tree for both position states", () => {
  assert.match(engine, /buildGravityTree\("Current"\);\s*run\(encoder, bindGroup, "integrateDrift"/);
  assert.match(engine, /run\(encoder, bindGroup, "integrateDrift"[\s\S]*buildGravityTree\("Drift"\);\s*run\(encoder, bindGroup, "integrateKick"/);
  assert.match(engine, /currentBody\.velocity \+= 0\.5 \* acceleration \* params\.dt/);
  assert.match(engine, /currentBody\.position \+= currentBody\.velocity \* params\.dt/);
  assert.match(engine, /currentBody\.velocity \+= 0\.5 \* treeAcceleration\(currentBody, index, true\) \* params\.dt/);
});

test("near-field gravity and contacts use exact local body interactions", () => {
  assert.match(engine, /let exactNearFieldRadius = max\(CELL_SIZE \* 1\.5, sourceBody\.radius \* 4\.0\)/);
  assert.match(engine, /nodeAabbDistance > exactNearFieldRadius/);
  assert.match(engine, /accumulateBodyGravity\(sourceBody, other, &acceleration\)/);
  assert.match(engine, /fn buildSpatialHash/);
  assert.match(engine, /fn hashCell/);
  assert.match(engine, /atomicExchange\(&metadata\[HASH_HEAD_OFFSET \+ bucket\]/);
  assert.match(engine, /fn selectPartners/);
  assert.match(engine, /run\(encoder, bindGroup, "clearMetadata", dispatchBodies\)/);
});

test("fragmentation atomically claims ordinary free slots", () => {
  assert.match(engine, /fn reserveGrowth/);
  assert.match(engine, /atomicCompareExchangeWeak\(&metadata\[SLOT_STATE_OFFSET \+ slot\], 0u, 1u\)/);
  assert.match(engine, /outputBodies\[claimed\] = Body/);
});

test("collisions use center-of-mass energy and gravitational binding", () => {
  assert.match(engine, /fn relativeKineticEnergy\(sourceBody: Body, other: Body\) -> f32/);
  assert.match(engine, /return 0\.5 \* reducedMass \* dot\(relativeVelocity, relativeVelocity\)/);
  assert.match(engine, /let centerOfMassEnergy = relativeKineticEnergy\(sourceBody, other\)/);
  assert.match(engine, /let bindingMagnitude = G \* sourceBody\.mass \* other\.mass \/ distance/);
  assert.match(engine, /let gravitationallyBound = postImpactEnergy - bindingMagnitude < 0\.0/);
  assert.match(engine, /let canMerge = !tidalBlocked[\s\S]*gravitationallyBound/);
  assert.doesNotMatch(engine, /MERGE_RATIO/);
});

test("bound merging preserves mass, linear momentum, and angular momentum", () => {
  assert.match(engine, /let center = \(sourceBody\.position \* sourceBody\.mass \+ other\.position \* other\.mass\) \/ totalMass/);
  assert.match(engine, /let centerVelocity = \(sourceBody\.velocity \* sourceBody\.mass \+ other\.velocity \* other\.mass\) \/ totalMass/);
  assert.match(engine, /fn pairAngularMomentum/);
  assert.match(engine, /sourceOrbital \+ otherOrbital[\s\S]*momentOfInertia\(sourceBody\) \* sourceBody\.spin/);
  assert.match(engine, /sourceBody\.mass = totalMass/);
  assert.match(engine, /sourceBody\.velocity = centerVelocity/);
  assert.match(engine, /sourceBody\.spin = totalAngularMomentum \/ max\(momentOfInertia\(sourceBody\), 0\.0001\)/);
});

test("non-merging contacts exchange normal and tangential impulses with spin", () => {
  assert.match(engine, /contactRelativeVelocity = relativeVelocity[\s\S]*sourceBody\.spin \* sourceBody\.radius/);
  assert.match(engine, /normalImpulse = -\(1\.0 \+ NORMAL_RESTITUTION\)/);
  assert.match(engine, /maximumTangentImpulse = CONTACT_FRICTION \* normalImpulse/);
  assert.match(engine, /sourceBody\.velocity -= impulse \/ sourceBody\.mass/);
  assert.match(engine, /sourceBody\.spin -= sourceBody\.radius \* tangentImpulse \/ sourceInertia/);
});

test("fragmentation responds to impact energy and conserves resolved invariants", () => {
  assert.match(engine, /let impactImpulse = reducedMass \* normalSpeed/);
  assert.match(engine, /energy \/ totalMass < disruptionThreshold\(totalMass\)/);
  assert.match(engine, /fragmentCount\(energy, totalMass, reducedMass, impactImpulse, obliquity\)/);
  assert.match(engine, /fn fragmentMass\(event: FragmentEvent, ordinal: u32, weightSum: f32\) -> f32/);
  assert.match(engine, /massWeightedPattern \+= fragmentPattern\(event, sample\) \* sampleMass/);
  assert.match(engine, /rawVelocityMean \+= rawVelocity \* sampleMass/);
  assert.match(engine, /rawAngularMomentum \+= cross2\(sampleOffset, centeredRawVelocity \* sampleMass\)/);
  assert.match(engine, /\(event\.angularMomentum - rawAngularMomentum\) \/ max\(rotationalInertia, 0\.0001\)/);
  assert.match(engine, /velocity = event\.centerVelocity[\s\S]*perpendicular\(positionOffset\) \* angularVelocity/);
  assert.doesNotMatch(engine, /0\.5 \* totalMass \* relativeSpeed \* relativeSpeed/);
});

test("fragment cascades remain bounded by mass resolution and generation", () => {
  assert.match(engine, /const MIN_FRAGMENT_MASS: f32 = 2\.0/);
  assert.match(engine, /const FRAGMENT_REARM_TIME: f32 = 1\.2/);
  assert.match(engine, /const MAX_FRAGMENT_GENERATION: f32 = 3\.0/);
  assert.match(engine, /let massLimitedCount = u32\(floor\(totalMass \/ MIN_FRAGMENT_MASS\)\)/);
  assert.match(engine, /if \(protectionActive \|\| generation >= MAX_FRAGMENT_GENERATION\) \{ return; \}/);
  assert.match(engine, /vec2<f32>\(params\.time \+ FRAGMENT_REARM_TIME, event\.generation \+ 1\.0\)/);
});

test("Roche limit tears loose bodies and blocks reaccretion in the tidal zone", () => {
  assert.match(engine, /const ROCHE_COEFFICIENT: f32 = 2\.44/);
  assert.match(engine, /fn rocheLimit\(primary: Body, satellite: Body\) -> f32/);
  assert.match(engine, /ROCHE_COEFFICIENT \* primary\.radius \* pow\(densityRatio, 1\.0 \/ 3\.0\)/);
  assert.match(engine, /fn strongestRochePrimary/);
  assert.match(engine, /fn detectRocheEvents/);
  assert.match(engine, /body\.tidalLockUntil = max\(body\.tidalLockUntil, params\.time \+ ROCHE_LOCK_TIME\)/);
  assert.match(engine, /storeSigned\(TIDAL_EVENT_OFFSET, index, i32\(eventIndex\)\)/);
  assert.match(engine, /let tidalBlocked = params\.time < max\(sourceBody\.tidalLockUntil, other\.tidalLockUntil\)/);
  assert.match(engine, /tangent \* \(-1\.5 \* event\.shearRate \* radialCoordinate\)/);
  assert.match(engine, /run\(encoder, bindGroup, "detectRocheEvents", dispatchBodies\)/);
});

test("physical surface accretion runs before gravity and overrides Roche locking", () => {
  assert.match(engine, /fn containingPrimary\(index: u32, sourceBody: Body\) -> i32/);
  assert.match(engine, /centerDistanceSquared < other\.radius \* other\.radius/);
  assert.match(engine, /other\.mass > primaryMass/);
  assert.match(engine, /fn detectSurfacePrimaries/);
  assert.match(engine, /fn accumulateSurfaceAccretion/);
  assert.match(engine, /fn resolveSurfaceAccretion/);
  assert.match(engine, /if \(loadSigned\(SURFACE_PRIMARY_OFFSET, index\) >= 0\) \{\s*outputBodies\[index\] = inactiveBody\(\)/);
  assert.match(engine, /let totalMass = originalMass \+ incomingMass/);
  assert.match(engine, /body\.velocity = totalMomentum \/ totalMass/);
  assert.match(engine, /totalGlobalAngularMomentum - cross2\(body\.position, totalMomentum\)/);
  assert.match(engine, /run\(encoder, surfaceBindGroup, "resolveSurfaceAccretion", dispatchBodies\)/);
  assert.match(engine, /run\(encoder, surfaceBindGroup, "syncSurfaceSlotState", dispatchBodies\)/);
  assert.match(engine, /const preAccretionCurrent = this\.currentBodies[\s\S]*const bindGroup = this\.createComputeBindGroup\(\)/);
  assert.doesNotMatch(engine, /queue\.submit\(\[surfaceEncoder\.finish\(\)\]\)/);
});

test("surface containment overrides Roche fragmentation and ordinary collision rules", () => {
  assert.match(engine, /fn accumulateBodyGravity[\s\S]*if \(physicalContainment\) \{\s*return;/);
  assert.match(engine, /if \(separation < primary\.radius\) \{\s*return;\s*\}/);
  assert.match(engine, /let surfaceContainment =[\s\S]*centerDistance < other\.radius[\s\S]*centerDistance < sourceBody\.radius/);
  assert.match(engine, /if \(surfaceContainment\) \{ return; \}/);
  assert.match(engine, /let sourceInsideOther = other\.mass > sourceBody\.mass && distance < other\.radius/);
  assert.match(engine, /let otherInsideSource = sourceBody\.mass > other\.mass && distance < sourceBody\.radius/);
  assert.match(engine, /if \(surfaceAccretion \|\| canMerge\)/);
  assert.match(engine, /if \(!surfaceAccretion\) \{[\s\S]*params\.lockedSlot/);
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

test("black hole creation uses compact radius with extreme mass", () => {
  assert.match(html, /data-value="black-hole"/);
  assert.match(html, /id="black-hole-controls"/);
  assert.match(main, /const BLACK_HOLE_RADIUS = 6/);
  assert.match(main, /const BLACK_HOLE_MASS = 5_000_000/);
  assert.match(main, /type CreationMode = "dynamic" \| "fixed" \| "black-hole"/);
  assert.match(main, /if \(mode === "black-hole"\) return BLACK_HOLE_RADIUS/);
  assert.match(main, /if \(mode === "black-hole"\) return BLACK_HOLE_MASS/);
  assert.match(main, /const mass = creationInjectionMass\(\)/);
});

test("compact bodies keep black-hole-like density when merging", () => {
  assert.match(engine, /const COMPACT_DENSITY_ENTER: f32 = DENSITY \* 64\.0/);
  assert.match(engine, /fn bodyDensity\(body: Body\) -> f32/);
  assert.match(engine, /fn mergedRadius\(sourceBody: Body, other: Body, totalMass: f32\) -> f32/);
  assert.match(engine, /let compactRadius = sqrt\(totalMass \/ inheritedDensity\)/);
  assert.match(engine, /let compactCollision = max\(bodyDensity\(sourceBody\), bodyDensity\(other\)\) >= COMPACT_DENSITY_ENTER/);
  assert.match(engine, /let nextRadius = mergedRadius\(sourceBody, other, totalMass\)/);
  assert.match(engine, /sourceBody\.radius = nextRadius/);
  assert.match(engine, /fn compactness\(body: Body\) -> f32/);
  assert.match(engine, /let compactColor = vec3<f32>\(0\.0, 0\.0, 0\.0\) \+ photonRing/);
});

test("camera pans with the right mouse button and suppresses its menu", () => {
  assert.match(main, /event\.button === 2 \|\| \(event\.button === 0 && spacePressed\)/);
  assert.doesNotMatch(main, /event\.button === 1/);
  assert.match(main, /addEventListener\("contextmenu", \(event\) => event\.preventDefault\(\)\)/);
});

test("UI exposes the compact three-panel simulation workspace", () => {
  assert.match(html, /<canvas id="space-canvas"><\/canvas>/);
  assert.match(html, /<canvas id="overlay-canvas"><\/canvas>/);
  assert.match(html, /<aside class="left-panel"/);
  assert.match(html, /<aside class="sidebar"/);
  assert.match(html, /<div class="command-bar">/);
  assert.doesNotMatch(html, /bottom-toolbar/);
  assert.match(html, /id="pause-button"/);
  assert.match(html, /id="btn-tool-create"/);
  assert.match(html, /id="btn-tool-select"/);
  assert.match(html, /id="timescale-value"/);
  assert.match(html, /id="camera-coords"/);
  assert.match(html, /id="creation-mode"/);
  assert.match(html, /id="inspect-panel"/);
  assert.match(main, /const overlayCanvas = document\.querySelector<HTMLCanvasElement>\("#overlay-canvas"\)!/);
  assert.match(main, /let activeTool: "create" \| "select"/);
  assert.doesNotMatch(html, /topbar|brand-mark|instruction|legend|camera-help|<footer/);
  assert.doesNotMatch(main, /mode-create|mode-select|notice/);
  assert.match(styles, /\.left-panel \{[\s\S]*?left: 0;[\s\S]*?width: var\(--panel-left-width\);/);
  assert.match(styles, /\.app-shell \{[\s\S]*?margin-right: var\(--panel-right-width\);[\s\S]*?padding: 0;/);
  assert.match(styles, /\.sidebar \{[\s\S]*?right: 0;[\s\S]*?width: var\(--panel-right-width\);/);
});

test("UI design system is compact, flat, and instrument-like", () => {
  assert.match(styles, /--bg-panel:/);
  assert.match(styles, /--border-subtle:/);
  assert.match(styles, /--text-muted:/);
  assert.match(styles, /--accent: #3b82f6/);
  assert.match(styles, /--space-1: 4px/);
  assert.match(styles, /font-variant-numeric: tabular-nums/);
  assert.match(styles, /\.command-bar \{/);
  assert.match(styles, /\.canvas-stage \{/);
  assert.match(styles, /\.body-card \{[\s\S]*?display: grid;/);
  assert.match(ui, /body-speed/);
  assert.match(ui, /this\.cards\.set\(body\.id, refs\)/);
  assert.match(ui, /body\.tidalLocked \? "Roche lock" : "Нет"/);
  assert.doesNotMatch(styles, /backdrop-filter|box-shadow|border-radius:\s*(?:[4-9]|1\d|2\d)px|border-radius:\s*50%/);
  assert.doesNotMatch(styles, /#4ade80/);
});

test("source text is valid UTF-8 without visible replacement mojibake", () => {
  for (const source of [html, main, ui, styles]) {
    assert.doesNotMatch(source, /�|пїЅ/);
  }
  assert.match(html, /Гравитационная симуляция/);
  assert.match(html, /Статус системы/);
  assert.match(main, /Продолжить/);
  assert.match(ui, /Выберите тело/);
});

test("mobile layout does not force a desktop minimum width", () => {
  const mobile = styles.slice(styles.indexOf("@media (max-width: 650px)"));
  assert.match(mobile, /min-width:\s*0/);
  assert.doesNotMatch(mobile, /min-width:\s*620px/);
});
