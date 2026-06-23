export type Vec2 = { x: number; y: number };
export type CameraState = { x: number; y: number; zoom: number };
export type CreationPreview = { position: Vec2; radius: number; vectorEnd?: Vec2 };

export type BodySnapshot = {
  id: number;
  name: string;
  position: Vec2;
  velocity: Vec2;
  mass: number;
  radius: number;
  hue: number;
};

type Injection = { position: Vec2; velocity: Vec2; radius: number; mass?: number };

export const BODY_COUNT = 4096;
const BODY_FLOATS = 8;
const BODY_BYTES = BODY_FLOATS * 4;
const WORKGROUP_SIZE = 64;
const HASH_BUCKET_COUNT = 8192;
const METADATA_WORDS = HASH_BUCKET_COUNT + BODY_COUNT * 4;
const MAX_FRAGMENT_EVENTS = BODY_COUNT / 2;
const MAX_FRAGMENTS = 15;
const GRID_ATTRACTOR_COUNT = 32;
const GRID_VERTEX_CAPACITY = 524288;
const MASS_DENSITY = 0.15;

const computeShader = /* wgsl */ `
  const BODY_COUNT: u32 = ${BODY_COUNT}u;
  const HASH_BUCKET_COUNT: u32 = ${HASH_BUCKET_COUNT}u;
  const MAX_FRAGMENT_EVENTS: u32 = ${MAX_FRAGMENT_EVENTS}u;
  const MAX_FRAGMENTS: u32 = ${MAX_FRAGMENTS}u;
  const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;
  const G: f32 = 9500.0;
  const SOFTENING: f32 = 12.0;
  const DENSITY: f32 = ${MASS_DENSITY};
  const MERGE_RATIO: f32 = 2.75;
  const FRAGMENT_SPEED: f32 = 62.0;
  const MIN_FRAGMENT_MASS: f32 = 2.0;
  const FRAGMENT_REARM_TIME: f32 = 1.2;
  const MAX_FRAGMENT_GENERATION: f32 = 2.0;
  const RESTITUTION: f32 = 0.35;
  const POSITION_CORRECTION: f32 = 0.5;
  const CELL_SIZE: f32 = 512.0;
  const HASH_HEAD_OFFSET: u32 = 0u;
  const HASH_NEXT_OFFSET: u32 = HASH_HEAD_OFFSET + HASH_BUCKET_COUNT;
  const CANDIDATE_OFFSET: u32 = HASH_NEXT_OFFSET + BODY_COUNT;
  const ACCEPTED_OFFSET: u32 = CANDIDATE_OFFSET + BODY_COUNT;
  const SLOT_STATE_OFFSET: u32 = ACCEPTED_OFFSET + BODY_COUNT;

  struct Body {
    position: vec2<f32>,
    velocity: vec2<f32>,
    mass: f32,
    radius: f32,
    // x: fragmentation protection deadline, y: resolved fragmentation generation.
    previous: vec2<f32>,
  };

  struct SimParams {
    dt: f32,
    time: f32,
    _pad: vec2<f32>,
  };

  struct FragmentEvent {
    center: vec2<f32>,
    centerVelocity: vec2<f32>,
    normal: vec2<f32>,
    sourceMass: f32,
    otherMass: f32,
    energy: f32,
    normalSpeed: f32,
    tangentSpeed: f32,
    obliquity: f32,
    count: u32,
    seed: u32,
    generation: f32,
    _pad: u32,
  };

  struct Counters {
    activeCount: atomic<u32>,
    events: atomic<u32>,
    reservedGrowth: atomic<u32>,
    maxRadiusBits: atomic<u32>,
  };

  @group(0) @binding(0) var<storage, read> currentBodies: array<Body>;
  @group(0) @binding(1) var<storage, read_write> driftBodies: array<Body>;
  @group(0) @binding(2) var<storage, read_write> kickBodies: array<Body>;
  @group(0) @binding(3) var<storage, read_write> outputBodies: array<Body>;
  @group(0) @binding(4) var<uniform> params: SimParams;
  @group(0) @binding(5) var<storage, read_write> metadata: array<atomic<u32>>;
  @group(0) @binding(6) var<storage, read_write> fragmentEvents: array<FragmentEvent>;
  @group(0) @binding(7) var<storage, read_write> counters: Counters;

  var<workgroup> bodyTile: array<Body, ${WORKGROUP_SIZE}>;

  fn inactiveBody() -> Body {
    return Body(vec2<f32>(0.0), vec2<f32>(0.0), 0.0, 0.0, vec2<f32>(0.0));
  }

  fn safeDirection(value: vec2<f32>, fallback: vec2<f32>) -> vec2<f32> {
    let magnitude = length(value);
    return select(fallback, value / magnitude, magnitude > 0.0001);
  }

  fn cellFor(position: vec2<f32>) -> vec2<i32> {
    return vec2<i32>(floor(position / CELL_SIZE));
  }

  fn hashCell(cell: vec2<i32>) -> u32 {
    let x = bitcast<u32>(cell.x) * 73856093u;
    let y = bitcast<u32>(cell.y) * 19349663u;
    return (x ^ y) & (HASH_BUCKET_COUNT - 1u);
  }

  fn loadSigned(offset: u32, index: u32) -> i32 {
    return bitcast<i32>(atomicLoad(&metadata[offset + index]));
  }

  fn storeSigned(offset: u32, index: u32, value: i32) {
    atomicStore(&metadata[offset + index], bitcast<u32>(value));
  }

  fn disruptionThreshold(totalMass: f32) -> f32 {
    // Larger aggregates need significantly more specific impact energy to disrupt,
    // scaling with square root of mass to approximate gravitational binding energy.
    return 0.125 * FRAGMENT_SPEED * FRAGMENT_SPEED * (1.0 + 0.20 * sqrt(totalMass));
  }

  fn fragmentCount(
    energy: f32,
    totalMass: f32,
    reducedMass: f32,
    impactImpulse: f32,
    obliquity: f32,
  ) -> u32 {
    let characteristicSpeed = sqrt(2.0 * energy / max(reducedMass, 0.0001));
    let energySeverity = max(0.0, characteristicSpeed / FRAGMENT_SPEED - 1.0);
    let impulseSeverity = impactImpulse / max(reducedMass * FRAGMENT_SPEED, 0.0001);
    let massCapacity = log2(1.0 + totalMass);
    let estimate = floor(
      3.0 + energySeverity * 3.2 + max(0.0, impulseSeverity - 0.35) * 1.2
      + massCapacity * 0.45 + obliquity * 1.25
    );
    return u32(clamp(estimate, 3.0, f32(MAX_FRAGMENTS)));
  }

  fn hash32(value: u32) -> u32 {
    var state = value;
    state ^= state >> 16u;
    state *= 0x7feb352du;
    state ^= state >> 15u;
    state *= 0x846ca68bu;
    state ^= state >> 16u;
    return state;
  }

  fn random01(seed: u32) -> f32 {
    return f32(hash32(seed)) / 4294967295.0;
  }

  fn reserveGrowth(growth: u32, freeSlots: u32) -> bool {
    var expected = atomicLoad(&counters.reservedGrowth);
    loop {
      if (expected + growth > freeSlots) { return false; }
      let result = atomicCompareExchangeWeak(&counters.reservedGrowth, expected, expected + growth);
      if (result.exchanged) { return true; }
      expected = result.old_value;
    }
  }

  @compute @workgroup_size(${WORKGROUP_SIZE})
  fn integrateDrift(
    @builtin(global_invocation_id) globalId: vec3<u32>,
    @builtin(local_invocation_id) localId: vec3<u32>,
  ) {
    let index = globalId.x;
    let valid = index < BODY_COUNT;
    var currentBody = inactiveBody();
    if (valid) { currentBody = currentBodies[index]; }
    let isActive = valid && currentBody.mass > 0.0;
    var acceleration = vec2<f32>(0.0);

    for (var tile = 0u; tile < BODY_COUNT; tile += WORKGROUP_SIZE) {
      let loadIndex = tile + localId.x;
      bodyTile[localId.x] = currentBodies[loadIndex];
      workgroupBarrier();
      if (isActive) {
        for (var offset = 0u; offset < WORKGROUP_SIZE; offset += 1u) {
          let otherIndex = tile + offset;
          let other = bodyTile[offset];
          if (other.mass > 0.0 && otherIndex != index) {
            let delta = other.position - currentBody.position;
            let distanceSquared = dot(delta, delta) + SOFTENING * SOFTENING;
            let inverseDistance = inverseSqrt(distanceSquared);
            acceleration += G * other.mass * delta * inverseDistance * inverseDistance * inverseDistance;
          }
        }
      }
      workgroupBarrier();
    }

    if (valid) {
      if (isActive) {
        currentBody.velocity += 0.5 * acceleration * params.dt;
        currentBody.position += currentBody.velocity * params.dt;
      }
      driftBodies[index] = currentBody;
    }
  }

  @compute @workgroup_size(${WORKGROUP_SIZE})
  fn integrateKick(
    @builtin(global_invocation_id) globalId: vec3<u32>,
    @builtin(local_invocation_id) localId: vec3<u32>,
  ) {
    let index = globalId.x;
    let valid = index < BODY_COUNT;
    var currentBody = inactiveBody();
    if (valid) { currentBody = driftBodies[index]; }
    let isActive = valid && currentBody.mass > 0.0;
    var acceleration = vec2<f32>(0.0);

    for (var tile = 0u; tile < BODY_COUNT; tile += WORKGROUP_SIZE) {
      let loadIndex = tile + localId.x;
      bodyTile[localId.x] = driftBodies[loadIndex];
      workgroupBarrier();
      if (isActive) {
        for (var offset = 0u; offset < WORKGROUP_SIZE; offset += 1u) {
          let otherIndex = tile + offset;
          let other = bodyTile[offset];
          if (other.mass > 0.0 && otherIndex != index) {
            let delta = other.position - currentBody.position;
            let distanceSquared = dot(delta, delta) + SOFTENING * SOFTENING;
            let inverseDistance = inverseSqrt(distanceSquared);
            acceleration += G * other.mass * delta * inverseDistance * inverseDistance * inverseDistance;
          }
        }
      }
      workgroupBarrier();
    }

    if (valid) {
      if (isActive) { currentBody.velocity += 0.5 * acceleration * params.dt; }
      kickBodies[index] = currentBody;
    }
  }

  @compute @workgroup_size(${WORKGROUP_SIZE})
  fn clearMetadata(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index < HASH_BUCKET_COUNT) { storeSigned(HASH_HEAD_OFFSET, index, -1); }
    if (index < BODY_COUNT) {
      storeSigned(HASH_NEXT_OFFSET, index, -1);
      storeSigned(CANDIDATE_OFFSET, index, -1);
      storeSigned(ACCEPTED_OFFSET, index, -1);
    }
    if (index == 0u) {
      atomicStore(&counters.activeCount, 0u);
      atomicStore(&counters.events, 0u);
      atomicStore(&counters.reservedGrowth, 0u);
      atomicStore(&counters.maxRadiusBits, 0u);
    }
  }

  @compute @workgroup_size(${WORKGROUP_SIZE})
  fn buildSpatialHash(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= BODY_COUNT) { return; }
    let body = kickBodies[index];
    if (body.mass <= 0.0) { return; }
    atomicAdd(&counters.activeCount, 1u);
    atomicMax(&counters.maxRadiusBits, bitcast<u32>(body.radius));
    let bucket = hashCell(cellFor(body.position));
    let previous = atomicExchange(&metadata[HASH_HEAD_OFFSET + bucket], index);
    storeSigned(HASH_NEXT_OFFSET, index, bitcast<i32>(previous));
  }

  fn deepestPartner(index: u32, sourceBody: Body) -> i32 {
    let ownCell = cellFor(sourceBody.position);
    let maximumRadius = bitcast<f32>(atomicLoad(&counters.maxRadiusBits));
    let cellReach = max(1, i32(ceil((sourceBody.radius + maximumRadius) / CELL_SIZE)));
    var partner = -1;
    var deepest = 0.0;
    for (var cellY = -cellReach; cellY <= cellReach; cellY += 1) {
      for (var cellX = -cellReach; cellX <= cellReach; cellX += 1) {
        let targetCell = ownCell + vec2<i32>(cellX, cellY);
        var cursor = loadSigned(HASH_HEAD_OFFSET, hashCell(targetCell));
        var guard = 0u;
        loop {
          if (cursor < 0 || guard >= BODY_COUNT) { break; }
          let otherIndex = u32(cursor);
          let other = kickBodies[otherIndex];
          if (otherIndex != index && other.mass > 0.0 && all(cellFor(other.position) == targetCell)) {
            let delta = other.position - sourceBody.position;
            let combinedRadius = sourceBody.radius + other.radius;
            if (abs(delta.x) <= combinedRadius && abs(delta.y) <= combinedRadius) {
              let distanceSquared = dot(delta, delta);
              if (distanceSquared < combinedRadius * combinedRadius) {
                let penetration = combinedRadius - sqrt(max(distanceSquared, 0.000001));
                if (penetration > deepest + 0.0001) {
                  deepest = penetration;
                  partner = cursor;
                }
              }
            }
          }
          cursor = loadSigned(HASH_NEXT_OFFSET, otherIndex);
          guard += 1u;
        }
      }
    }
    return partner;
  }

  @compute @workgroup_size(${WORKGROUP_SIZE})
  fn selectPartners(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= BODY_COUNT || kickBodies[index].mass <= 0.0) { return; }
    storeSigned(CANDIDATE_OFFSET, index, deepestPartner(index, kickBodies[index]));
  }

  @compute @workgroup_size(${WORKGROUP_SIZE})
  fn detectFragmentEvents(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= BODY_COUNT) { return; }
    let partnerValue = loadSigned(CANDIDATE_OFFSET, index);
    if (partnerValue < 0) { return; }
    let partner = u32(partnerValue);
    if (partner <= index || loadSigned(CANDIDATE_OFFSET, partner) != i32(index)) { return; }

    let sourceBody = kickBodies[index];
    let other = kickBodies[partner];
    let genA = select(0.0, sourceBody.previous.y, params.time < sourceBody.previous.x);
    let genB = select(0.0, other.previous.y, params.time < other.previous.x);
    let generation = max(genA, genB);
    let protectionActive = params.time < max(sourceBody.previous.x, other.previous.x);
    if (protectionActive || generation >= MAX_FRAGMENT_GENERATION) { return; }
    let normal = safeDirection(other.position - sourceBody.position, vec2<f32>(1.0, 0.0));
    let tangent = vec2<f32>(-normal.y, normal.x);
    let relativeVelocity = other.velocity - sourceBody.velocity;
    let relativeSpeed = length(relativeVelocity);
    let totalMass = sourceBody.mass + other.mass;
    let reducedMass = sourceBody.mass * other.mass / totalMass;
    let normalSpeed = max(0.0, -dot(relativeVelocity, normal));
    let tangentSpeed = dot(relativeVelocity, tangent);
    // Tangential motion couples less efficiently than the closing component.
    let energy = 0.5 * reducedMass * (normalSpeed * normalSpeed + 0.35 * tangentSpeed * tangentSpeed);
    let obliquity = abs(tangentSpeed) / max(relativeSpeed, 0.0001);
    if (normalSpeed < FRAGMENT_SPEED * 0.05 || energy / totalMass < disruptionThreshold(totalMass)) { return; }

    let impactImpulse = reducedMass * normalSpeed;
    let requestedCount = fragmentCount(energy, totalMass, reducedMass, impactImpulse, obliquity);
    let massLimitedCount = u32(floor(totalMass / MIN_FRAGMENT_MASS));
    let count = min(requestedCount, massLimitedCount);
    if (count < 3u) { return; }
    let freeSlots = BODY_COUNT - atomicLoad(&counters.activeCount);
    let growth = select(0u, count - 2u, count > 2u);
    if (!reserveGrowth(growth, freeSlots)) { return; }

    let eventIndex = atomicAdd(&counters.events, 1u);
    if (eventIndex >= MAX_FRAGMENT_EVENTS) { return; }
    fragmentEvents[eventIndex] = FragmentEvent(
      (sourceBody.position * sourceBody.mass + other.position * other.mass) / totalMass,
      (sourceBody.velocity * sourceBody.mass + other.velocity * other.mass) / totalMass,
      normal,
      sourceBody.mass,
      other.mass,
      energy,
      normalSpeed,
      tangentSpeed,
      obliquity,
      count,
      hash32((index * 73856093u) ^ (partner * 19349663u) ^ u32(params.time * 1000.0)),
      generation,
      0u,
    );
    storeSigned(ACCEPTED_OFFSET, index, i32(partner));
    storeSigned(ACCEPTED_OFFSET, partner, i32(index));
  }

  @compute @workgroup_size(${WORKGROUP_SIZE})
  fn resolveCollisions(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= BODY_COUNT) { return; }
    var sourceBody = kickBodies[index];
    let partnerValue = loadSigned(CANDIDATE_OFFSET, index);
    if (sourceBody.mass <= 0.0 || partnerValue < 0) {
      outputBodies[index] = sourceBody;
      return;
    }
    let partner = u32(partnerValue);
    if (loadSigned(CANDIDATE_OFFSET, partner) != i32(index)) {
      outputBodies[index] = sourceBody;
      return;
    }
    let other = kickBodies[partner];
    if (loadSigned(ACCEPTED_OFFSET, index) == i32(partner)) {
      outputBodies[index] = inactiveBody();
      return;
    }

    let ratio = max(sourceBody.mass, other.mass) / min(sourceBody.mass, other.mass);
    let relativeVelocity = other.velocity - sourceBody.velocity;
    let normal = safeDirection(other.position - sourceBody.position, vec2<f32>(1.0, 0.0));
    let tangent = vec2<f32>(-normal.y, normal.x);
    let totalMass = sourceBody.mass + other.mass;
    let reducedMass = sourceBody.mass * other.mass / totalMass;
    let normalSpeed = max(0.0, -dot(relativeVelocity, normal));
    let tangentSpeed = dot(relativeVelocity, tangent);
    let coupledEnergy = 0.5 * reducedMass * (normalSpeed * normalSpeed + 0.35 * tangentSpeed * tangentSpeed);
    let disruptive = normalSpeed >= FRAGMENT_SPEED * 0.05
      && coupledEnergy / totalMass >= disruptionThreshold(totalMass);
    let protectionActive = params.time < max(sourceBody.previous.x, other.previous.x);
    let genA = select(0.0, sourceBody.previous.y, params.time < sourceBody.previous.x);
    let genB = select(0.0, other.previous.y, params.time < other.previous.x);
    let generation = max(genA, genB);
    let unresolvedCollision = min(sourceBody.mass, other.mass) < MIN_FRAGMENT_MASS * 1.5;
    let survivor = sourceBody.mass > other.mass || (abs(sourceBody.mass - other.mass) < 0.0001 && index < partner);
    if (ratio >= MERGE_RATIO || disruptive || protectionActive || generation >= MAX_FRAGMENT_GENERATION || unresolvedCollision) {
      if (!survivor) {
        outputBodies[index] = inactiveBody();
        return;
      }
      sourceBody.position = (sourceBody.position * sourceBody.mass + other.position * other.mass) / totalMass;
      sourceBody.velocity = (sourceBody.velocity * sourceBody.mass + other.velocity * other.mass) / totalMass;
      sourceBody.mass = totalMass;
      sourceBody.radius = sqrt(totalMass / DENSITY);
      let mergedGeneration = (sourceBody.mass * genA + other.mass * genB) / totalMass;
      sourceBody.previous = vec2<f32>(
        max(sourceBody.previous.x, other.previous.x),
        mergedGeneration,
      );
      outputBodies[index] = sourceBody;
      return;
    }

    let delta = other.position - sourceBody.position;
    let fallback = select(vec2<f32>(-1.0, 0.0), vec2<f32>(1.0, 0.0), index < partner);
    let contactNormal = safeDirection(delta, fallback);
    let distance = max(length(delta), 0.001);
    let overlap = max(0.0, sourceBody.radius + other.radius - distance);
    sourceBody.position -= contactNormal * overlap * (other.mass / (sourceBody.mass + other.mass)) * POSITION_CORRECTION;
    let signedNormalSpeed = dot(relativeVelocity, contactNormal);
    if (signedNormalSpeed < 0.0) {
      let impulse = -(1.0 + RESTITUTION) * signedNormalSpeed / (1.0 / sourceBody.mass + 1.0 / other.mass);
      sourceBody.velocity -= contactNormal * impulse / sourceBody.mass;
    }
    outputBodies[index] = sourceBody;
  }

  @compute @workgroup_size(${WORKGROUP_SIZE})
  fn syncSlotState(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= BODY_COUNT) { return; }
    atomicStore(&metadata[SLOT_STATE_OFFSET + index], select(0u, 1u, outputBodies[index].mass > 0.0));
  }

  fn fragmentWeight(event: FragmentEvent, ordinal: u32) -> f32 {
    let totalMass = event.sourceMass + event.otherMass;
    let reducedMass = event.sourceMass * event.otherMass / totalMass;
    let severity = max(0.0, sqrt(2.0 * event.energy / max(reducedMass, 0.0001)) / FRAGMENT_SPEED - 1.0);
    if (ordinal == 0u) {
      return max(0.15, event.sourceMass / totalMass) * 4.0 / (1.0 + 0.7 * severity);
    }
    if (ordinal == 1u) {
      return max(0.15, event.otherMass / totalMass) * 4.0 / (1.0 + 0.7 * severity);
    }
    let variation = 0.25 + 0.75 * random01(event.seed ^ (ordinal * 2654435761u));
    return variation * (0.7 + 0.5 * severity);
  }

  fn fragmentPattern(event: FragmentEvent, ordinal: u32) -> vec2<f32> {
    let phase = random01(event.seed ^ 0xa511e9b3u) * 6.28318530718;
    let jitter = (random01(event.seed ^ (ordinal * 2246822519u) ^ 0x9e3779b9u) - 0.5) * 0.7;
    let angle = phase + 2.39996322973 * f32(ordinal) + jitter;
    let relativeSpeed = sqrt(event.normalSpeed * event.normalSpeed + event.tangentSpeed * event.tangentSpeed);
    let headOn = event.normalSpeed / max(relativeSpeed, 0.0001);
    let tangent = vec2<f32>(-event.normal.y, event.normal.x);
    let normalScale = 0.55 + 1.2 * headOn;
    let tangentScale = 0.55 + 1.45 * event.obliquity;
    let rawDirection = event.normal * cos(angle) * normalScale + tangent * sin(angle) * tangentScale;
    let speedVariation = 0.65 + 0.7 * random01(event.seed ^ (ordinal * 3266489917u) ^ 0x85ebca6bu);
    return safeDirection(rawDirection, event.normal) * speedVariation;
  }

  @compute @workgroup_size(${WORKGROUP_SIZE})
  fn spawnFragments(@builtin(global_invocation_id) id: vec3<u32>) {
    let flatIndex = id.x;
    let eventIndex = flatIndex / MAX_FRAGMENTS;
    let ordinal = flatIndex % MAX_FRAGMENTS;
    let eventCount = min(atomicLoad(&counters.events), MAX_FRAGMENT_EVENTS);
    if (eventIndex >= eventCount) { return; }
    let event = fragmentEvents[eventIndex];
    if (ordinal >= event.count) { return; }

    let start = (eventIndex * 2053u + ordinal * 977u + u32(params.time * 1000.0)) % BODY_COUNT;
    var claimed = BODY_COUNT;
    for (var attempt = 0u; attempt < BODY_COUNT; attempt += 1u) {
      let slot = (start + attempt) % BODY_COUNT;
      let result = atomicCompareExchangeWeak(&metadata[SLOT_STATE_OFFSET + slot], 0u, 1u);
      if (result.exchanged) {
        claimed = slot;
        break;
      }
    }
    if (claimed >= BODY_COUNT) { return; }

    let totalMass = event.sourceMass + event.otherMass;
    var weightSum = 0.0;
    var weightedPattern = vec2<f32>(0.0);
    for (var sample = 0u; sample < event.count; sample += 1u) {
      let sampleWeight = fragmentWeight(event, sample);
      weightSum += sampleWeight;
      weightedPattern += fragmentPattern(event, sample) * sampleWeight;
    }
    let meanPattern = weightedPattern / max(weightSum, 0.0001);
    let weight = fragmentWeight(event, ordinal);
    let pattern = fragmentPattern(event, ordinal) - meanPattern;
    let distributableMass = max(0.0, totalMass - MIN_FRAGMENT_MASS * f32(event.count));
    let mass = MIN_FRAGMENT_MASS + distributableMass * weight / weightSum;
    let radius = sqrt(mass / DENSITY);

    var weightedVariance = 0.0;
    for (var sample = 0u; sample < event.count; sample += 1u) {
      let centered = fragmentPattern(event, sample) - meanPattern;
      weightedVariance += fragmentWeight(event, sample) * dot(centered, centered);
    }
    let patternEnergyMass = totalMass * weightedVariance / max(weightSum, 0.0001);
    let severity = event.energy / max(totalMass * disruptionThreshold(totalMass), 0.0001);
    let ejectaFraction = clamp(0.12 + 0.07 * severity + 0.10 * event.obliquity, 0.12, 0.48);
    let speedScale = sqrt(2.0 * event.energy * ejectaFraction / max(patternEnergyMass, 0.0001));
    let parentRadius = sqrt(totalMass / DENSITY);
    let spread = parentRadius * (1.15 + 0.18 * sqrt(f32(event.count)));
    // The mass-weighted pattern mean is zero, so both center of mass and
    // linear momentum remain at the pre-impact barycentric values.
    let position = event.center + pattern * spread;
    let velocity = event.centerVelocity + pattern * speedScale;
    outputBodies[claimed] = Body(
      position,
      velocity,
      mass,
      radius,
      vec2<f32>(params.time + FRAGMENT_REARM_TIME, event.generation + 1.0),
    );
  }
`;

const mutationShader = /* wgsl */ `
  const BODY_COUNT: u32 = ${BODY_COUNT}u;
  const SLOT_STATE_OFFSET: u32 = ${HASH_BUCKET_COUNT + BODY_COUNT * 3}u;
  struct Body {
    position: vec2<f32>, velocity: vec2<f32>, mass: f32, radius: f32, previous: vec2<f32>,
  };
  struct Mutation { header: vec4<f32>, values: vec4<f32> };
  @group(0) @binding(0) var<storage, read_write> bodies: array<Body>;
  @group(0) @binding(1) var<storage, read_write> metadata: array<atomic<u32>>;
  @group(0) @binding(2) var<uniform> mutation: Mutation;

  @compute @workgroup_size(1)
  fn mutate() {
    let kind = u32(mutation.header.x);
    if (kind == 1u) {
      let start = u32(mutation.header.y) % BODY_COUNT;
      for (var attempt = 0u; attempt < BODY_COUNT; attempt += 1u) {
        let slot = (start + attempt) % BODY_COUNT;
        let result = atomicCompareExchangeWeak(&metadata[SLOT_STATE_OFFSET + slot], 0u, 1u);
        if (result.exchanged) {
          let position = mutation.values.xy;
          bodies[slot] = Body(
            position,
            mutation.values.zw,
            mutation.header.w,
            mutation.header.z,
            vec2<f32>(0.0),
          );
          return;
        }
      }
    } else if (kind == 2u) {
      let slot = u32(mutation.header.y);
      if (slot < BODY_COUNT) {
        bodies[slot] = Body(vec2<f32>(0.0), vec2<f32>(0.0), 0.0, 0.0, vec2<f32>(0.0));
        atomicStore(&metadata[SLOT_STATE_OFFSET + slot], 0u);
      }
    }
  }
`;

const renderShader = /* wgsl */ `
  const BODY_COUNT: u32 = ${BODY_COUNT}u;
  const GRID_ATTRACTOR_COUNT: u32 = ${GRID_ATTRACTOR_COUNT}u;
  struct Body {
    position: vec2<f32>, velocity: vec2<f32>, mass: f32, radius: f32, previous: vec2<f32>,
  };
  struct ViewUniforms {
    view: vec4<f32>,
    state: vec4<f32>,
    preview: vec4<f32>,
    vector: vec4<f32>,
  };
  @group(0) @binding(0) var<storage, read> bodies: array<Body>;
  @group(0) @binding(1) var<uniform> uniforms: ViewUniforms;
  @group(0) @binding(2) var<storage, read> gridAttractorIndices: array<u32>;

  fn worldToClip(world: vec2<f32>) -> vec2<f32> {
    let screen = (world - uniforms.view.xy) * uniforms.view.z;
    return vec2<f32>(screen.x / (uniforms.view.w * 0.5), -screen.y / (uniforms.state.x * 0.5));
  }

  fn heatColor(mass: f32, speed: f32) -> vec3<f32> {
    let t = 1.0 - exp(-(mass * 0.012 + speed * 0.005));
    let cold = vec3<f32>(0.39, 0.41, 0.45);
    let warm = vec3<f32>(0.71, 0.12, 0.08);
    let orange = vec3<f32>(1.0, 0.45, 0.12);
    let hot = vec3<f32>(1.0, 0.84, 0.39);
    if (t < 0.35) { return mix(cold, warm, t / 0.35); }
    if (t < 0.70) { return mix(warm, orange, (t - 0.35) / 0.35); }
    return mix(orange, hot, (t - 0.70) / 0.30);
  }

  struct BodyVertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) selected: f32,
    @location(3) bodyVisible: f32,
  };

  @vertex
  fn bodyVertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> BodyVertexOut {
    let corners = array<vec2<f32>, 6>(
      vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
      vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    let body = bodies[instance];
    let local = corners[vertex];
    var out: BodyVertexOut;
    out.position = vec4<f32>(worldToClip(body.position + local * body.radius), 0.0, 1.0);
    if (body.mass <= 0.0) { out.position = vec4<f32>(2.0, 2.0, 0.0, 1.0); }
    out.local = local;
    out.color = heatColor(body.mass, length(body.velocity));
    out.selected = select(0.0, 1.0, abs(f32(instance) - uniforms.state.z) < 0.5);
    out.bodyVisible = select(0.0, 1.0, body.mass > 0.0);
    return out;
  }

  @fragment
  fn bodyFragment(input: BodyVertexOut) -> @location(0) vec4<f32> {
    let radiusSquared = dot(input.local, input.local);
    if (radiusSquared > 1.0 || input.bodyVisible < 0.5) { discard; }
    let edge = 1.0;
    let highlight = 1.0 - smoothstep(0.0, 0.46, length(input.local + vec2<f32>(0.28, 0.30)));
    var color = input.color * (0.72 + highlight * 0.45);
    var alpha = edge;
    if (input.selected > 0.5) {
      let ring = smoothstep(0.52, 0.7, radiusSquared);
      color = mix(color, vec3<f32>(0.55, 0.86, 1.0), ring * 0.8);
      alpha = max(alpha, ring);
    }
    return vec4<f32>(color, alpha);
  }

  struct LineOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
  };

  @vertex
  fn trailVertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> LineOut {
    let body = bodies[instance];
    let tail = body.position - body.velocity * 0.11;
    let point = select(tail, body.position, vertex == 1u);
    var out: LineOut;
    out.position = vec4<f32>(worldToClip(point), 0.0, 1.0);
    if (body.mass <= 0.0) { out.position = vec4<f32>(2.0, 2.0, 0.0, 1.0); }
    out.color = vec4<f32>(heatColor(body.mass, length(body.velocity)), select(0.08, 0.42, vertex == 1u));
    return out;
  }

  @fragment fn lineFragment(input: LineOut) -> @location(0) vec4<f32> { return input.color; }

  struct GridInput {
    @location(0) position: vec2<f32>,
    @location(1) opacity: f32,
  };

  @vertex
  fn gridVertex(input: GridInput) -> LineOut {
    var world = input.position;
    var potential = 0.0;
    for (var index = 0u; index < GRID_ATTRACTOR_COUNT; index += 1u) {
      if (index >= u32(uniforms.state.w)) { break; }
      // Only the selection is updated by the CPU. Body state is read from the
      // live simulation buffer, so grid deformation follows every GPU step.
      let attractor = bodies[gridAttractorIndices[index]];
      if (attractor.mass > 0.0) {
        let towardBody = attractor.position - world;
        let distance = length(towardBody);
        let direction = towardBody / max(distance, 0.0001);
        let depth = attractor.mass * 55.0;
        let softening = attractor.radius * 2.0 + 40.0;
        let rawPull = depth / (distance + softening);
        // The exponential map can approach full compression without ever
        // pulling a vertex through the attractor and folding the grid.
        let compression = 1.0 - exp(-rawPull / max(distance, 0.0001));
        world += direction * distance * compression;
        potential += attractor.mass / (distance + softening);
      }
    }
    let wellStrength = 1.0 - exp(-potential * 0.18);
    let color = mix(vec3<f32>(0.15, 0.15, 0.15), vec3<f32>(0.65, 0.65, 0.65), wellStrength * 0.70);
    var out: LineOut;
    out.position = vec4<f32>(worldToClip(world), 0.0, 1.0);
    out.color = vec4<f32>(color, input.opacity * (1.0 + wellStrength * 0.45));
    return out;
  }

  @vertex
  fn previewVertex(@builtin(vertex_index) vertex: u32) -> BodyVertexOut {
    let corners = array<vec2<f32>, 6>(
      vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
      vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    let local = corners[vertex];
    var out: BodyVertexOut;
    out.position = vec4<f32>(worldToClip(uniforms.preview.xy + local * uniforms.preview.z), 0.0, 1.0);
    if (uniforms.preview.w < 0.5) { out.position = vec4<f32>(2.0, 2.0, 0.0, 1.0); }
    out.local = local;
    out.color = vec3<f32>(0.55, 0.86, 1.0);
    out.selected = 0.0;
    out.bodyVisible = uniforms.preview.w;
    return out;
  }

  @fragment
  fn previewFragment(input: BodyVertexOut) -> @location(0) vec4<f32> {
    let distance = length(input.local);
    if (distance > 1.0 || input.bodyVisible < 0.5) { discard; }
    let ring = 1.0 - smoothstep(0.89, 1.0, distance);
    let inner = smoothstep(0.0, 0.9, distance);
    return vec4<f32>(input.color, 0.12 + ring * inner * 0.65);
  }

  @vertex
  fn vectorVertex(@builtin(vertex_index) vertex: u32) -> LineOut {
    let point = select(uniforms.preview.xy, uniforms.vector.xy, vertex == 1u);
    var out: LineOut;
    out.position = vec4<f32>(worldToClip(point), 0.0, 1.0);
    if (uniforms.vector.z < 0.5) { out.position = vec4<f32>(2.0, 2.0, 0.0, 1.0); }
    out.color = vec4<f32>(0.55, 0.86, 1.0, 0.85);
    return out;
  }
`;

export class GPUEngine {
  readonly capacity = BODY_COUNT;

  private readonly canvas: HTMLCanvasElement;
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private currentBodies: GPUBuffer;
  private driftBodies: GPUBuffer;
  private kickBodies: GPUBuffer;
  private outputBodies: GPUBuffer;
  private readonly simParams: GPUBuffer;
  private readonly renderUniforms: GPUBuffer;
  private readonly gridAttractorIndices: GPUBuffer;
  private readonly mutationUniform: GPUBuffer;
  private readonly metadata: GPUBuffer;
  private readonly fragmentEvents: GPUBuffer;
  private readonly counters: GPUBuffer;
  private readonly snapshotBuffer: GPUBuffer;
  private readonly computeLayout: GPUBindGroupLayout;
  private readonly renderLayout: GPUBindGroupLayout;
  private readonly mutationLayout: GPUBindGroupLayout;
  private readonly computePipelines: Record<string, GPUComputePipeline>;
  private readonly mutationPipeline: GPUComputePipeline;
  private readonly bodyPipeline: GPURenderPipeline;
  private readonly trailPipeline: GPURenderPipeline;
  private readonly gridPipeline: GPURenderPipeline;
  private readonly previewPipeline: GPURenderPipeline;
  private readonly vectorPipeline: GPURenderPipeline;
  private gridBuffer: GPUBuffer;
  private gridVertexCapacity = GRID_VERTEX_CAPACITY;
  private gridVertexCount = 0;
  private renderBindGroup: GPUBindGroup;
  private readonly injectionQueue: Injection[] = [];
  private readonly deletionQueue: number[] = [];
  private readonly occupied = new Uint8Array(BODY_COUNT);
  private lastSnapshots: BodySnapshot[] = [];
  private readonly lastCamera: CameraState = { x: 0, y: 0, zoom: 1 };
  private readonly attractorCamera: CameraState = { x: Number.NaN, y: Number.NaN, zoom: Number.NaN };
  private readonly gridGeometryCamera: CameraState = { x: Number.NaN, y: Number.NaN, zoom: Number.NaN };
  private gridGeometryWidth = 0;
  private gridGeometryHeight = 0;
  private gridWarpDepth = 0;
  private gridWarpSoftening = 40;
  private gridGeometryWarpDepth = -1;
  private gridGeometryWarpSoftening = Number.POSITIVE_INFINITY;
  private lastAttractorUpdate = 0;
  private gridAttractorCount = 0;
  private snapshotInFlight = false;
  private selectedSlot = -1;
  private width = 1;
  private height = 1;
  private elapsed = 0;
  private mutationSeed = 0;

  static async create(canvas: HTMLCanvasElement): Promise<GPUEngine> {
    if (!navigator.gpu) throw new Error("Для симуляции требуется браузер с поддержкой WebGPU.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("Не удалось получить WebGPU-адаптер.");
    const device = await adapter.requestDevice();
    device.addEventListener("uncapturederror", (event) => {
      console.error("WebGPU validation error:", event.error.message);
    });
    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("Не удалось создать WebGPU-контекст.");
    device.pushErrorScope("validation");
    const engine = new GPUEngine(canvas, device, context);
    const validationError = await device.popErrorScope();
    if (validationError) throw new Error(`WebGPU pipeline validation failed: ${validationError.message}`);
    return engine;
  }

  private constructor(canvas: HTMLCanvasElement, device: GPUDevice, context: GPUCanvasContext) {
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: "premultiplied" });

    const bodyUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.currentBodies = this.createBuffer(BODY_COUNT * BODY_BYTES, bodyUsage, "body-current");
    this.driftBodies = this.createBuffer(BODY_COUNT * BODY_BYTES, bodyUsage, "body-drift");
    this.kickBodies = this.createBuffer(BODY_COUNT * BODY_BYTES, bodyUsage, "body-kick");
    this.outputBodies = this.createBuffer(BODY_COUNT * BODY_BYTES, bodyUsage, "body-output");
    this.simParams = this.createBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "sim-params");
    this.renderUniforms = this.createBuffer(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "view-uniforms");
    this.gridAttractorIndices = this.createBuffer(
      GRID_ATTRACTOR_COUNT * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      "grid-attractor-indices",
    );
    this.mutationUniform = this.createBuffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "mutation");
    this.metadata = this.createBuffer(METADATA_WORDS * 4, GPUBufferUsage.STORAGE, "simulation-metadata");
    this.fragmentEvents = this.createBuffer(MAX_FRAGMENT_EVENTS * 64, GPUBufferUsage.STORAGE, "fragment-events");
    this.counters = this.createBuffer(16, GPUBufferUsage.STORAGE, "counters");
    this.snapshotBuffer = this.createBuffer(
      BODY_COUNT * BODY_BYTES,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      "snapshot-staging",
    );

    this.computeLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const computeModule = device.createShaderModule({ code: computeShader, label: "gravity-compute" });
    const computePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
    const entries = [
      "integrateDrift", "integrateKick", "clearMetadata", "buildSpatialHash", "selectPartners",
      "detectFragmentEvents", "resolveCollisions", "syncSlotState", "spawnFragments",
    ];
    this.computePipelines = Object.fromEntries(entries.map((entryPoint) => [
      entryPoint,
      device.createComputePipeline({ layout: computePipelineLayout, compute: { module: computeModule, entryPoint } }),
    ]));

    this.mutationLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    const mutationModule = device.createShaderModule({ code: mutationShader, label: "body-mutation" });
    this.mutationPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.mutationLayout] }),
      compute: { module: mutationModule, entryPoint: "mutate" },
    });

    this.renderLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
    ] });
    const renderModule = device.createShaderModule({ code: renderShader, label: "gravity-render" });
    const renderPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.renderLayout] });
    const blend: GPUBlendState = {
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    };
    const target = { format: this.format, blend };
    this.bodyPipeline = device.createRenderPipeline({
      layout: renderPipelineLayout,
      vertex: { module: renderModule, entryPoint: "bodyVertex" },
      fragment: { module: renderModule, entryPoint: "bodyFragment", targets: [target] },
      primitive: { topology: "triangle-list" },
    });
    this.trailPipeline = device.createRenderPipeline({
      layout: renderPipelineLayout,
      vertex: { module: renderModule, entryPoint: "trailVertex" },
      fragment: { module: renderModule, entryPoint: "lineFragment", targets: [target] },
      primitive: { topology: "line-list" },
    });
    this.gridPipeline = device.createRenderPipeline({
      layout: renderPipelineLayout,
      vertex: {
        module: renderModule,
        entryPoint: "gridVertex",
        buffers: [{
          arrayStride: 12,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x2" },
            { shaderLocation: 1, offset: 8, format: "float32" },
          ],
        }],
      },
      fragment: { module: renderModule, entryPoint: "lineFragment", targets: [target] },
      primitive: { topology: "line-list" },
    });
    this.previewPipeline = device.createRenderPipeline({
      layout: renderPipelineLayout,
      vertex: { module: renderModule, entryPoint: "previewVertex" },
      fragment: { module: renderModule, entryPoint: "previewFragment", targets: [target] },
      primitive: { topology: "triangle-list" },
    });
    this.vectorPipeline = device.createRenderPipeline({
      layout: renderPipelineLayout,
      vertex: { module: renderModule, entryPoint: "vectorVertex" },
      fragment: { module: renderModule, entryPoint: "lineFragment", targets: [target] },
      primitive: { topology: "line-list" },
    });

    this.gridBuffer = device.createBuffer({
      size: GRID_VERTEX_CAPACITY * 12,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      label: "dynamic-grid",
    });

    this.renderBindGroup = this.createRenderBindGroup();
  }

  private createBuffer(size: number, usage: GPUBufferUsageFlags, label: string): GPUBuffer {
    return this.device.createBuffer({ size, usage, label });
  }

  private createComputeBindGroup(): GPUBindGroup {
    const buffers = [
      this.currentBodies, this.driftBodies, this.kickBodies, this.outputBodies, this.simParams,
      this.metadata, this.fragmentEvents, this.counters,
    ];
    return this.device.createBindGroup({
      layout: this.computeLayout,
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  }

  private createRenderBindGroup(): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.renderLayout,
      entries: [
        { binding: 0, resource: { buffer: this.currentBodies } },
        { binding: 1, resource: { buffer: this.renderUniforms } },
        { binding: 2, resource: { buffer: this.gridAttractorIndices } },
      ],
    });
  }

  private updateGridAttractorIndices(force = false): void {
    const now = performance.now();
    const zoomRatio = this.lastCamera.zoom / this.attractorCamera.zoom;
    const cameraShiftPixels = Math.hypot(
      this.lastCamera.x - this.attractorCamera.x,
      this.lastCamera.y - this.attractorCamera.y,
    ) * this.lastCamera.zoom;
    if (!force && now - this.lastAttractorUpdate < 120 && cameraShiftPixels < 40 && zoomRatio > 0.9 && zoomRatio < 1.1) {
      return;
    }

    const viewRadius = Math.max(1, Math.hypot(this.width, this.height) * 0.5 / this.lastCamera.zoom);
    const strongest = this.lastSnapshots
      .map((body) => {
        const distance = Math.hypot(body.position.x - this.lastCamera.x, body.position.y - this.lastCamera.y);
        return { body, score: body.mass / (1 + distance / viewRadius) };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, GRID_ATTRACTOR_COUNT);

    const data = new Uint32Array(GRID_ATTRACTOR_COUNT);
    let warpDepth = 0;
    let warpSoftening = Number.POSITIVE_INFINITY;
    strongest.forEach(({ body }, index) => {
      data[index] = body.id;
      warpDepth += body.mass * 55;
      warpSoftening = Math.min(warpSoftening, body.radius * 2 + 40);
    });
    this.gridAttractorCount = strongest.length;
    this.gridWarpDepth = warpDepth;
    this.gridWarpSoftening = Number.isFinite(warpSoftening) ? warpSoftening : 40;
    this.device.queue.writeBuffer(this.gridAttractorIndices, 0, data.buffer as ArrayBuffer);
    this.attractorCamera.x = this.lastCamera.x;
    this.attractorCamera.y = this.lastCamera.y;
    this.attractorCamera.zoom = this.lastCamera.zoom;
    this.lastAttractorUpdate = now;
  }

  private gridLod(zoom: number): number {
    const desiredStep = 80 / Math.max(zoom, 0.000001);
    const decade = 10 ** Math.floor(Math.log10(desiredStep));
    const normalized = desiredStep / decade;
    let fineStep: number;
    let coarseStep: number;
    let progress: number;
    if (normalized < 2) {
      fineStep = decade;
      coarseStep = decade * 2;
      progress = (normalized - 1) / (2 - 1);
    } else if (normalized < 5) {
      fineStep = decade * 2;
      coarseStep = decade * 5;
      progress = (normalized - 2) / (5 - 2);
    } else {
      fineStep = decade * 5;
      coarseStep = decade * 10;
      progress = (normalized - 5) / (10 - 5);
    }
    const clamped = Math.min(1, Math.max(0, progress));
    const blend = clamped * clamped * (3 - 2 * clamped);
    // A single continuously changing step keeps every neighbouring line
    // equally spaced and avoids the dense/sparse bands caused by two grids.
    return fineStep * (coarseStep / fineStep) ** blend;
  }

  private updateGridGeometry(camera: CameraState): void {
    if (
      camera.x === this.gridGeometryCamera.x &&
      camera.y === this.gridGeometryCamera.y &&
      camera.zoom === this.gridGeometryCamera.zoom &&
      this.width === this.gridGeometryWidth &&
      this.height === this.gridGeometryHeight &&
      this.gridWarpDepth <= this.gridGeometryWarpDepth &&
      this.gridWarpSoftening >= this.gridGeometryWarpSoftening
    ) {
      return;
    }
    const gridStep = this.gridLod(camera.zoom);
    const worldPerPixel = 1 / camera.zoom;
    const viewWidth = this.width * worldPerPixel;
    const viewHeight = this.height * worldPerPixel;
    // Ограничения силы деформации нет. Вместо фиксированного запаса решаем
    // консервативную границу M = base + depth / (M + softening), после которой
    // даже суммарное притяжение выбранных масс не втянет край mesh в viewport.
    const warpMargin = (base: number): number => {
      const coefficient = base + this.gridWarpSoftening;
      const displacement = (
        -coefficient + Math.sqrt(coefficient * coefficient + 4 * this.gridWarpDepth)
      ) * 0.5;
      return base + Math.max(0, displacement) * 1.5;
    };
    const marginX = warpMargin(96 * worldPerPixel);
    const marginY = warpMargin(96 * worldPerPixel);
    const minX = camera.x - viewWidth * 0.5 - marginX;
    const maxX = camera.x + viewWidth * 0.5 + marginX;
    const minY = camera.y - viewHeight * 0.5 - marginY;
    const maxY = camera.y + viewHeight * 0.5 + marginY;
    const verticalSegments = Math.max(1, Math.min(96, Math.ceil((maxY - minY) / (28 * worldPerPixel))));
    const horizontalSegments = Math.max(1, Math.min(96, Math.ceil((maxX - minX) / (28 * worldPerPixel))));
    const values: number[] = [];

    const appendGrid = (step: number, opacity: number): void => {
      if (opacity < 0.001) return;
      const firstX = Math.floor(minX / step) * step;
      for (let x = firstX; x <= maxX; x += step) {
        for (let segment = 0; segment < verticalSegments; segment += 1) {
          const y0 = minY + (maxY - minY) * segment / verticalSegments;
          const y1 = minY + (maxY - minY) * (segment + 1) / verticalSegments;
          values.push(x, y0, opacity, x, y1, opacity);
        }
      }
      const firstY = Math.floor(minY / step) * step;
      for (let y = firstY; y <= maxY; y += step) {
        for (let segment = 0; segment < horizontalSegments; segment += 1) {
          const x0 = minX + (maxX - minX) * segment / horizontalSegments;
          const x1 = minX + (maxX - minX) * (segment + 1) / horizontalSegments;
          values.push(x0, y, opacity, x1, y, opacity);
        }
      }
    };

    appendGrid(gridStep, 0.18);

    // Оси рисуются поверх LOD и также получают гравитационную деформацию.
    if (minX <= 0 && maxX >= 0) {
      for (let segment = 0; segment < verticalSegments; segment += 1) {
        const y0 = minY + (maxY - minY) * segment / verticalSegments;
        const y1 = minY + (maxY - minY) * (segment + 1) / verticalSegments;
        values.push(0, y0, 0.24, 0, y1, 0.24);
      }
    }
    if (minY <= 0 && maxY >= 0) {
      for (let segment = 0; segment < horizontalSegments; segment += 1) {
        const x0 = minX + (maxX - minX) * segment / horizontalSegments;
        const x1 = minX + (maxX - minX) * (segment + 1) / horizontalSegments;
        values.push(x0, 0, 0.24, x1, 0, 0.24);
      }
    }

    const vertices = new Float32Array(values);
    this.gridVertexCount = vertices.length / 3;
    if (this.gridVertexCount > this.gridVertexCapacity) {
      while (this.gridVertexCapacity < this.gridVertexCount) this.gridVertexCapacity *= 2;
      const requiredBytes = this.gridVertexCapacity * 12;
      if (requiredBytes > this.device.limits.maxBufferSize) {
        throw new Error(`Dynamic grid requires ${requiredBytes} bytes, exceeding the GPU buffer limit`);
      }
      this.gridBuffer.destroy();
      this.gridBuffer = this.device.createBuffer({
        size: requiredBytes,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        label: "dynamic-grid",
      });
    }
    if (this.gridVertexCount > 0) {
      const byteLength = this.gridVertexCount * 12;
      this.device.queue.writeBuffer(this.gridBuffer, 0, vertices.buffer as ArrayBuffer, 0, byteLength);
    }
    this.gridGeometryCamera.x = camera.x;
    this.gridGeometryCamera.y = camera.y;
    this.gridGeometryCamera.zoom = camera.zoom;
    this.gridGeometryWidth = this.width;
    this.gridGeometryHeight = this.height;
    this.gridGeometryWarpDepth = this.gridWarpDepth;
    this.gridGeometryWarpSoftening = this.gridWarpSoftening;
  }

  private applyMutation(kind: number, slot: number, injection?: Injection): void {
    const radius = injection?.radius ?? 0;
    const mass = injection?.mass ?? (radius * radius * MASS_DENSITY);
    const data = new Float32Array([
      kind, slot, radius, mass,
      injection?.position.x ?? 0, injection?.position.y ?? 0,
      injection?.velocity.x ?? 0, injection?.velocity.y ?? 0,
    ]);
    this.device.queue.writeBuffer(this.mutationUniform, 0, data);
    const bindGroup = this.device.createBindGroup({
      layout: this.mutationLayout,
      entries: [
        { binding: 0, resource: { buffer: this.currentBodies } },
        { binding: 1, resource: { buffer: this.metadata } },
        { binding: 2, resource: { buffer: this.mutationUniform } },
      ],
    });
    const encoder = this.device.createCommandEncoder({ label: "body-mutation" });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.mutationPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  injectBody(position: Vec2, velocity: Vec2, radius: number, mass?: number): number | null {
    const estimatedActive = this.occupied.reduce((sum, value) => sum + value, 0) + this.injectionQueue.length;
    if (estimatedActive >= BODY_COUNT) return null;
    this.injectionQueue.push({ position: { ...position }, velocity: { ...velocity }, radius, mass });
    return 0;
  }

  deleteBody(slot: number): void {
    if (slot >= 0 && slot < BODY_COUNT) this.deletionQueue.push(slot);
  }

  setSelected(slot: number): void {
    this.selectedSlot = slot;
  }

  resize(width: number, height: number): void {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    const physicalWidth = Math.max(1, Math.floor(this.width * pixelRatio));
    const physicalHeight = Math.max(1, Math.floor(this.height * pixelRatio));
    if (this.canvas.width !== physicalWidth || this.canvas.height !== physicalHeight) {
      this.canvas.width = physicalWidth;
      this.canvas.height = physicalHeight;
    }
  }

  step(dt: number): void {
    this.flushPendingMutations();
    this.elapsed += dt;
    this.device.queue.writeBuffer(this.simParams, 0, new Float32Array([dt, this.elapsed, 0, 0]));
    const bindGroup = this.createComputeBindGroup();
    const encoder = this.device.createCommandEncoder({ label: "gravity-step" });
    const dispatchBodies = Math.ceil(BODY_COUNT / WORKGROUP_SIZE);

    const run = (name: keyof typeof this.computePipelines, groups: number): void => {
      const pass = encoder.beginComputePass({ label: name });
      pass.setPipeline(this.computePipelines[name]);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(groups);
      pass.end();
    };

    run("integrateDrift", dispatchBodies);
    run("integrateKick", dispatchBodies);
    run("clearMetadata", Math.ceil(HASH_BUCKET_COUNT / WORKGROUP_SIZE));
    run("buildSpatialHash", dispatchBodies);
    run("selectPartners", dispatchBodies);
    run("detectFragmentEvents", dispatchBodies);
    run("resolveCollisions", dispatchBodies);
    run("syncSlotState", dispatchBodies);
    run("spawnFragments", Math.ceil(MAX_FRAGMENT_EVENTS * MAX_FRAGMENTS / WORKGROUP_SIZE));
    this.device.queue.submit([encoder.finish()]);

    const oldCurrent = this.currentBodies;
    this.currentBodies = this.outputBodies;
    this.outputBodies = this.kickBodies;
    this.kickBodies = this.driftBodies;
    this.driftBodies = oldCurrent;
    this.renderBindGroup = this.createRenderBindGroup();
  }

  flushPendingMutations(): boolean {
    const deletion = this.deletionQueue.shift();
    if (deletion !== undefined) {
      this.applyMutation(2, deletion);
      this.occupied[deletion] = 0;
      return true;
    }
    const injection = this.injectionQueue.shift();
    if (injection) {
      this.mutationSeed = (this.mutationSeed + 977) % BODY_COUNT;
      this.applyMutation(1, this.mutationSeed, injection);
      return true;
    }
    return false;
  }

  render(camera: CameraState, preview: CreationPreview | null, showGrid = true): void {
    while (this.flushPendingMutations()) {}
    this.lastCamera.x = camera.x;
    this.lastCamera.y = camera.y;
    this.lastCamera.zoom = camera.zoom;
    this.updateGridAttractorIndices();
    if (showGrid) {
      this.updateGridGeometry(camera);
    }
    const vector = preview?.vectorEnd;
    const uniforms = new Float32Array([
      camera.x, camera.y, camera.zoom, this.width,
      this.height, this.elapsed, this.selectedSlot, this.gridAttractorCount,
      preview?.position.x ?? 0, preview?.position.y ?? 0, preview?.radius ?? 0, preview ? 1 : 0,
      vector?.x ?? 0, vector?.y ?? 0, vector ? 1 : 0, 0,
    ]);
    this.device.queue.writeBuffer(this.renderUniforms, 0, uniforms);
    const encoder = this.device.createCommandEncoder({ label: "gravity-render" });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass.setBindGroup(0, this.renderBindGroup);
    if (showGrid) {
      pass.setPipeline(this.gridPipeline);
      pass.setVertexBuffer(0, this.gridBuffer);
      pass.draw(this.gridVertexCount);
    }
    pass.setPipeline(this.trailPipeline);
    pass.draw(2, BODY_COUNT);
    pass.setPipeline(this.bodyPipeline);
    pass.draw(6, BODY_COUNT);
    if (preview) {
      pass.setPipeline(this.previewPipeline);
      pass.draw(6);
      if (vector) {
        pass.setPipeline(this.vectorPipeline);
        pass.draw(2);
      }
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  async readSnapshot(): Promise<BodySnapshot[]> {
    if (this.snapshotInFlight) return this.lastSnapshots;
    this.snapshotInFlight = true;
    try {
      const encoder = this.device.createCommandEncoder({ label: "body-snapshot" });
      encoder.copyBufferToBuffer(this.currentBodies, 0, this.snapshotBuffer, 0, BODY_COUNT * BODY_BYTES);
      this.device.queue.submit([encoder.finish()]);
      await this.snapshotBuffer.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(this.snapshotBuffer.getMappedRange().slice(0));
      this.snapshotBuffer.unmap();
      this.occupied.fill(0);
      const snapshots: BodySnapshot[] = [];
      for (let slot = 0; slot < BODY_COUNT; slot += 1) {
        const offset = slot * BODY_FLOATS;
        const mass = values[offset + 4];
        if (!(mass > 0)) continue;
        this.occupied[slot] = 1;
        const vx = values[offset + 2];
        const vy = values[offset + 3];
        const heat = 1 - Math.exp(-(mass * 0.012 + Math.hypot(vx, vy) * 0.005));
        snapshots.push({
          id: slot,
          name: `Небесное тело ${slot + 1}`,
          position: { x: values[offset], y: values[offset + 1] },
          velocity: { x: vx, y: vy },
          mass,
          radius: values[offset + 5],
          hue: Math.round(15 + heat * 200),
        });
      }
      this.lastSnapshots = snapshots;
      this.updateGridAttractorIndices(true);
      return snapshots;
    } finally {
      this.snapshotInFlight = false;
    }
  }
}
