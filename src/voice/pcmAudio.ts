/**
 * Pure WebAudio 16kHz PCM capture (MicStreamer) and 24kHz PCM playback (PcmPlayer).
 * Replaces Agora RTC audio with native browser WebAudio APIs.
 */

const WORKLET_PROCESSOR_CODE = `
class PcmCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length > 0) {
      this.port.postMessage(input[0]);
    }
    return true;
  }
}
registerProcessor('pcm-capture-processor', PcmCaptureProcessor);
`;

export class MicStreamer {
  private audioCtx: AudioContext;
  private ownContext = false;
  private mediaStream: MediaStream | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private scriptNode: ScriptProcessorNode | null = null;
  private muted = false;
  private started = false;

  private onChunkCb: ((chunk: ArrayBuffer) => void) | null = null;
  private onLevelCb: ((level: number) => void) | null = null;

  // Resampling state
  private inputBuffer: Float32Array = new Float32Array(0);
  private readonly targetRate = 16000;
  private readonly chunkSize = 512; // ~32ms at 16kHz (1024 bytes)
  private chunkBuffer = new Int16Array(this.chunkSize);
  private chunkIndex = 0;
  private sumSquares = 0;
  private samplesInChunk = 0;

  constructor(audioCtx?: AudioContext) {
    if (audioCtx) {
      this.audioCtx = audioCtx;
    } else {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.audioCtx = new AudioCtx();
      this.ownContext = true;
    }
  }

  onChunk(cb: (chunk: ArrayBuffer) => void) {
    this.onChunkCb = cb;
  }

  onLevel(cb: (level: number) => void) {
    this.onLevelCb = cb;
  }

  setMuted(muted: boolean) {
    this.muted = muted;
    if (this.mediaStream) {
      this.mediaStream.getAudioTracks().forEach((track) => {
        track.enabled = !muted;
      });
    }
    if (muted && this.onLevelCb) {
      this.onLevelCb(0);
    }
  }

  isMuted(): boolean {
    return this.muted;
  }

  async start(): Promise<void> {
    if (this.started) return;

    if (this.audioCtx.state === 'suspended') {
      try {
        await this.audioCtx.resume();
      } catch {
        /* noop */
      }
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    this.mediaStream = stream;
    if (this.muted) {
      stream.getAudioTracks().forEach((track) => {
        track.enabled = false;
      });
    }

    const source = this.audioCtx.createMediaStreamSource(stream);
    this.sourceNode = source;

    let useWorklet = false;
    if (this.audioCtx.audioWorklet && typeof AudioWorkletNode !== 'undefined') {
      try {
        const blob = new Blob([WORKLET_PROCESSOR_CODE], { type: 'application/javascript' });
        const blobUrl = URL.createObjectURL(blob);
        try {
          await this.audioCtx.audioWorklet.addModule(blobUrl);
          const worklet = new AudioWorkletNode(this.audioCtx, 'pcm-capture-processor');
          worklet.port.onmessage = (event: MessageEvent<Float32Array>) => {
            this.handleAudioFrame(event.data);
          };
          source.connect(worklet);
          this.workletNode = worklet;
          useWorklet = true;
        } finally {
          URL.revokeObjectURL(blobUrl);
        }
      } catch (err) {
        console.warn('[medsim:pcm] AudioWorklet init failed, falling back to ScriptProcessor:', err);
      }
    }

    if (!useWorklet) {
      // Fallback: ScriptProcessorNode
      const scriptNode = this.audioCtx.createScriptProcessor(2048, 1, 1);
      scriptNode.onaudioprocess = (e: AudioProcessingEvent) => {
        const inputData = e.inputBuffer.getChannelData(0);
        this.handleAudioFrame(inputData);
      };
      source.connect(scriptNode);
      // Connect to destination via silent gain to ensure processing in some browsers
      const silentGain = this.audioCtx.createGain();
      silentGain.gain.value = 0;
      scriptNode.connect(silentGain);
      silentGain.connect(this.audioCtx.destination);
      this.scriptNode = scriptNode;
    }

    this.started = true;
  }

  private handleAudioFrame(input: Float32Array): void {
    if (this.muted || !this.started) {
      this.onLevelCb?.(0);
      return;
    }

    const inputRate = this.audioCtx.sampleRate;
    const ratio = inputRate / this.targetRate;

    // Append new input data to any leftover samples
    const combined = new Float32Array(this.inputBuffer.length + input.length);
    combined.set(this.inputBuffer, 0);
    combined.set(input, this.inputBuffer.length);

    let srcPos = 0;
    while (srcPos + ratio < combined.length) {
      const idx = Math.floor(srcPos);
      const frac = srcPos - idx;
      const s0 = combined[idx];
      const s1 = combined[idx + 1] ?? s0;
      const sample = s0 + frac * (s1 - s0);
      srcPos += ratio;

      // Clamp to Int16 range
      const intSample = Math.max(-32768, Math.min(32767, Math.round(sample * 32767)));
      this.chunkBuffer[this.chunkIndex++] = intSample;
      this.sumSquares += sample * sample;
      this.samplesInChunk++;

      if (this.chunkIndex >= this.chunkSize) {
        // Compute RMS level (0..1)
        const rms = Math.sqrt(this.sumSquares / this.samplesInChunk);
        const level = Math.min(1, rms * 4.5);
        this.onLevelCb?.(level);

        // Copy chunk buffer and emit
        const chunkToSend = this.chunkBuffer.slice(0, this.chunkSize).buffer;
        this.onChunkCb?.(chunkToSend);

        // Reset chunk
        this.chunkIndex = 0;
        this.sumSquares = 0;
        this.samplesInChunk = 0;
      }
    }

    // Keep leftover samples for next frame
    const remaining = combined.length - Math.floor(srcPos);
    if (remaining > 0 && srcPos < combined.length) {
      this.inputBuffer = combined.slice(Math.floor(srcPos));
    } else {
      this.inputBuffer = new Float32Array(0);
    }
  }

  stop(): void {
    this.started = false;
    if (this.workletNode) {
      try { this.workletNode.disconnect(); } catch { /* noop */ }
      this.workletNode = null;
    }
    if (this.scriptNode) {
      try { this.scriptNode.disconnect(); } catch { /* noop */ }
      this.scriptNode = null;
    }
    if (this.sourceNode) {
      try { this.sourceNode.disconnect(); } catch { /* noop */ }
      this.sourceNode = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => {
        try { track.stop(); } catch { /* noop */ }
      });
      this.mediaStream = null;
    }
    if (this.ownContext) {
      try { this.audioCtx.close(); } catch { /* noop */ }
    }
    this.inputBuffer = new Float32Array(0);
    this.chunkIndex = 0;
    this.sumSquares = 0;
    this.samplesInChunk = 0;
  }
}

export class PcmPlayer {
  private audioCtx: AudioContext;
  private sampleRate: number;
  private analyser: AnalyserNode;
  private gainNode: GainNode;
  private nextPlayTime = 0;
  private activeSources = new Set<AudioBufferSourceNode>();
  private levelData: Uint8Array;
  private smoothedAmp = 0;
  private closed = false;

  constructor(audioCtx: AudioContext, sampleRate = 24000) {
    this.audioCtx = audioCtx;
    this.sampleRate = sampleRate;

    this.analyser = this.audioCtx.createAnalyser();
    this.analyser.fftSize = 256;
    this.levelData = new Uint8Array(this.analyser.frequencyBinCount);

    this.gainNode = this.audioCtx.createGain();
    this.gainNode.gain.value = 1.0;

    this.analyser.connect(this.gainNode);
    this.gainNode.connect(this.audioCtx.destination);
  }

  enqueue(pcmBuffer: ArrayBuffer): void {
    if (this.closed || pcmBuffer.byteLength < 2) return;

    if (this.audioCtx.state === 'suspended') {
      try {
        void this.audioCtx.resume();
      } catch {
        /* noop */
      }
    }

    const sampleCount = Math.floor(pcmBuffer.byteLength / 2);
    const int16 = new Int16Array(pcmBuffer, 0, sampleCount);
    const float32 = new Float32Array(sampleCount);

    for (let i = 0; i < sampleCount; i++) {
      float32[i] = int16[i] / 32768;
    }

    const audioBuffer = this.audioCtx.createBuffer(1, sampleCount, this.sampleRate);
    audioBuffer.copyToChannel(float32, 0);

    const source = this.audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.analyser);

    const now = this.audioCtx.currentTime;
    // Add small buffer margin (20ms) if cursor has fallen behind current time
    const startTime = Math.max(now + 0.02, this.nextPlayTime);
    source.start(startTime);
    this.nextPlayTime = startTime + audioBuffer.duration;

    this.activeSources.add(source);
    source.onended = () => {
      this.activeSources.delete(source);
    };
  }

  /**
   * Instantly stops all scheduled audio playback and resets the play cursor.
   * Essential for immediate barge-in when the agent or user interrupts.
   */
  clear(): void {
    for (const source of this.activeSources) {
      try {
        source.stop();
        source.disconnect();
      } catch {
        /* noop */
      }
    }
    this.activeSources.clear();
    this.nextPlayTime = 0;
    this.smoothedAmp = 0;
  }

  /**
   * Reads playback RMS level (0..1) for lip-sync and talking indicators.
   */
  getOutputLevel(): number {
    if (this.closed || this.activeSources.size === 0) {
      this.smoothedAmp = 0;
      return 0;
    }

    (this.analyser.getByteTimeDomainData as (arr: Uint8Array) => void)(this.levelData);
    let sumSquares = 0;
    for (let i = 0; i < this.levelData.length; i++) {
      const normalized = (this.levelData[i] - 128) / 128;
      sumSquares += normalized * normalized;
    }

    const rms = Math.sqrt(sumSquares / this.levelData.length);
    // Scale RMS into responsive 0..1 mouth factor
    const target = Math.min(1, rms * 3.6);
    this.smoothedAmp = this.smoothedAmp * 0.45 + target * 0.55;
    return this.smoothedAmp < 0.02 ? 0 : this.smoothedAmp;
  }

  close(): void {
    this.closed = true;
    this.clear();
    try { this.analyser.disconnect(); } catch { /* noop */ }
    try { this.gainNode.disconnect(); } catch { /* noop */ }
  }
}
