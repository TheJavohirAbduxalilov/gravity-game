export class AudioManager {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private musicGain: GainNode | null = null;
  private noiseBuffer: AudioBuffer | null = null;

  // Background drones
  private osc1: OscillatorNode | null = null;
  private osc2: OscillatorNode | null = null;
  private osc3: OscillatorNode | null = null;
  private musicIntervalId: any = null;

  // Creation nodes
  private creationOsc: OscillatorNode | null = null;
  private creationGain: GainNode | null = null;

  constructor() {}

  private init(): void {
    if (this.ctx) return;
    try {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      this.ctx = new AudioContextClass();

      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = 0.65;
      this.masterGain.connect(this.ctx.destination);

      this.musicGain = this.ctx.createGain();
      this.musicGain.gain.value = 0.05; // Soft ambient volume
      this.musicGain.connect(this.masterGain);

      this.noiseBuffer = this.createNoiseBuffer(this.ctx);

      this.startBackgroundMusic();
    } catch (e) {
      console.warn("Failed to initialize Web Audio API:", e);
    }
  }

  resume(): void {
    this.init();
    if (this.ctx && this.ctx.state === "suspended") {
      void this.ctx.resume();
    }
  }

  private createNoiseBuffer(ctx: AudioContext): AudioBuffer {
    const bufferSize = ctx.sampleRate * 2.0;
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = Math.random() * 2 - 1;
    }
    return buffer;
  }

  private startBackgroundMusic(): void {
    if (!this.ctx || !this.musicGain) return;

    // Create 3 slow sine oscillators for a spacey chord drone
    const now = this.ctx.currentTime;
    
    this.osc1 = this.ctx.createOscillator();
    this.osc2 = this.ctx.createOscillator();
    this.osc3 = this.ctx.createOscillator();

    this.osc1.type = "sine";
    this.osc2.type = "sine";
    this.osc3.type = "sine";

    const notes = [
      [55.0, 110.0, 164.8], // A1, A2, E3 (Am drone)
      [43.6, 87.3, 130.8],  // F1, F2, C3 (F drone)
      [65.4, 130.8, 196.0], // C2, C3, G3 (C drone)
      [49.0, 98.0, 146.8]   // G1, G2, D3 (G drone)
    ];

    let chordIndex = 0;
    const initialChord = notes[chordIndex];

    this.osc1.frequency.setValueAtTime(initialChord[0], now);
    this.osc2.frequency.setValueAtTime(initialChord[1], now);
    this.osc3.frequency.setValueAtTime(initialChord[2], now);

    const g1 = this.ctx.createGain();
    const g2 = this.ctx.createGain();
    const g3 = this.ctx.createGain();

    g1.gain.value = 0.45;
    g2.gain.value = 0.35;
    g3.gain.value = 0.25;

    this.osc1.connect(g1).connect(this.musicGain);
    this.osc2.connect(g2).connect(this.musicGain);
    this.osc3.connect(g3).connect(this.musicGain);

    this.osc1.start(now);
    this.osc2.start(now);
    this.osc3.start(now);

    const playNextChord = () => {
      if (!this.ctx || this.ctx.state === "suspended") return;
      chordIndex = (chordIndex + 1) % notes.length;
      const nextChord = notes[chordIndex];
      const time = this.ctx.currentTime;

      this.osc1?.frequency.exponentialRampToValueAtTime(nextChord[0], time + 5.0);
      this.osc2?.frequency.exponentialRampToValueAtTime(nextChord[1], time + 5.0);
      this.osc3?.frequency.exponentialRampToValueAtTime(nextChord[2], time + 5.0);
    };

    this.musicIntervalId = setInterval(playNextChord, 12000);
  }

  startCreation(): void {
    this.resume();
    if (!this.ctx || !this.masterGain) return;

    // Terminate existing creation oscillator if active
    if (this.creationOsc) {
      try { this.creationOsc.stop(); } catch (e) {}
    }

    const now = this.ctx.currentTime;
    this.creationOsc = this.ctx.createOscillator();
    this.creationGain = this.ctx.createGain();

    this.creationOsc.type = "sine";
    this.creationOsc.frequency.setValueAtTime(80, now);
    this.creationOsc.frequency.exponentialRampToValueAtTime(380, now + 4.5);

    this.creationGain.gain.setValueAtTime(0.001, now);
    this.creationGain.gain.linearRampToValueAtTime(0.12, now + 0.15);

    this.creationOsc.connect(this.creationGain).connect(this.masterGain);
    this.creationOsc.start(now);
  }

  updateCreationPitch(radius: number): void {
    if (!this.ctx || !this.creationOsc) return;
    // Map radius to oscillator pitch
    const targetFreq = Math.min(750, 80 + radius * 5.5);
    this.creationOsc.frequency.setTargetAtTime(targetFreq, this.ctx.currentTime, 0.12);
  }

  stopCreation(speedMagnitude: number): void {
    if (!this.ctx || !this.creationOsc || !this.creationGain) return;

    const now = this.ctx.currentTime;
    const currentFreq = this.creationOsc.frequency.value;
    this.creationOsc.frequency.setValueAtTime(currentFreq, now);
    this.creationGain.gain.setValueAtTime(this.creationGain.gain.value, now);

    if (speedMagnitude > 2.0) {
      // Rapid sweep down (whoosh)
      const duration = 0.35;
      this.creationOsc.frequency.exponentialRampToValueAtTime(Math.max(35, currentFreq * 0.18), now + duration);
      this.creationGain.gain.exponentialRampToValueAtTime(0.001, now + duration);
      
      const osc = this.creationOsc;
      setTimeout(() => {
        try { osc.stop(); } catch (e) {}
      }, duration * 1000 + 50);
    } else {
      // Fade out
      this.creationGain.gain.exponentialRampToValueAtTime(0.001, now + 0.1);
      const osc = this.creationOsc;
      setTimeout(() => {
        try { osc.stop(); } catch (e) {}
      }, 150);
    }

    this.creationOsc = null;
    this.creationGain = null;
  }

  playCollision(mass: number, factor: number, zoom: number): void {
    this.resume();
    if (!this.ctx || !this.masterGain || factor <= 0.05) return;

    const now = this.ctx.currentTime;
    const zoomFactor = Math.min(1.0, zoom);
    const volume = Math.min(0.7, Math.sqrt(mass) * 0.075 * factor * zoomFactor);

    // Cinematic planetary collision: deep rumbles, seismic tremors, and dark crust fractures.
    const duration = Math.min(3.5, 1.0 + Math.sqrt(mass) * 0.1);
    const baseFreq = Math.max(22, 55 - Math.sqrt(mass) * 0.15); // extremely deep sub-bass (22Hz - 55Hz)

    // 1. Gravitational Shockwave: Two slightly detuned, deep oscillators for massive acoustic churning
    const sub1 = this.ctx.createOscillator();
    const sub2 = this.ctx.createOscillator();
    const subGain = this.ctx.createGain();

    sub1.type = "sine";
    sub1.frequency.setValueAtTime(baseFreq, now);
    sub1.frequency.linearRampToValueAtTime(baseFreq * 0.8, now + duration);

    sub2.type = "sine";
    sub2.frequency.setValueAtTime(baseFreq * 1.04, now); // Detune slightly to create slow acoustic beating
    sub2.frequency.linearRampToValueAtTime(baseFreq * 1.04 * 0.8, now + duration);

    subGain.gain.setValueAtTime(0.001, now);
    subGain.gain.linearRampToValueAtTime(volume * 0.7, now + 0.08); // Slower attack (80ms) to convey colossal mass/scale
    subGain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    sub1.connect(subGain);
    sub2.connect(subGain);
    subGain.connect(this.masterGain);

    sub1.start(now);
    sub2.start(now);
    sub1.stop(now + duration + 0.1);
    sub2.stop(now + duration + 0.1);

    // 2. Crust Crack / Impact Fracture: A dark, muddy noise burst for the initial planetary split
    if (this.noiseBuffer) {
      const crackle = this.ctx.createBufferSource();
      crackle.buffer = this.noiseBuffer;

      const crackleFilter = this.ctx.createBiquadFilter();
      crackleFilter.type = "bandpass";
      crackleFilter.frequency.setValueAtTime(220, now);
      crackleFilter.Q.value = 1.2; // Low Q for a muddy, non-synthesized shatter sound

      const crackleGain = this.ctx.createGain();
      crackleGain.gain.setValueAtTime(volume * 0.45, now);
      crackleGain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);

      crackle.connect(crackleFilter).connect(crackleGain).connect(this.masterGain);
      crackle.start(now);
      crackle.stop(now + 0.4);
    }

    // 3. Tectonic Tremor & Rock Grinding: sustaining lowpass noise modulated by an LFO (tremolo)
    if (this.noiseBuffer) {
      const grind = this.ctx.createBufferSource();
      grind.buffer = this.noiseBuffer;
      grind.loop = true;

      const grindFilter = this.ctx.createBiquadFilter();
      grindFilter.type = "lowpass";
      // Filter out high-frequency hiss, keeping only the heavy low-mid mud
      const cutoff = Math.max(45, 180 - Math.sqrt(mass) * 0.4);
      grindFilter.frequency.setValueAtTime(cutoff, now);

      const grindGain = this.ctx.createGain();
      grindGain.gain.setValueAtTime(0.001, now);
      grindGain.gain.linearRampToValueAtTime(volume * 0.55, now + 0.15); // Slow rumble build-up
      grindGain.gain.exponentialRampToValueAtTime(0.001, now + duration);

      // Amplitude Modulation (LFO): oscillates at 10-14 Hz to create rolling earthquake vibrations
      const lfo = this.ctx.createOscillator();
      const lfoGain = this.ctx.createGain();
      
      lfo.type = "sine";
      lfo.frequency.setValueAtTime(10 + Math.random() * 4, now);

      lfoGain.gain.setValueAtTime(volume * 0.22, now);
      lfoGain.gain.exponentialRampToValueAtTime(0.001, now + duration);

      // Connect LFO -> lfoGain -> grindGain.gain
      lfo.connect(lfoGain).connect(grindGain.gain);
      
      grind.connect(grindFilter).connect(grindGain).connect(this.masterGain);
      
      lfo.start(now);
      grind.start(now);

      lfo.stop(now + duration + 0.1);
      grind.stop(now + duration + 0.1);
    }
  }
}
