/**
 * Faculty side of the live observer. Connects to the consultation's voice
 * channel via native WebSocket to /voice/observer/ws?channel={channel}&uid={uid}&token={token}.
 *
 * Plays student (16kHz PCM, prefix 0x01) and AI patient (24kHz PCM, prefix 0x02)
 * audio via PcmPlayer instances, reads live transcripts and chart events,
 * and sends whispers and marks to the student.
 */

import { parseAgentMessage, TranscriptAssembler, type TranscriptLine } from '../voice/transcriptParser';
import { PcmPlayer } from '../voice/pcmAudio';
import {
  encodeMedsim,
  isForMe,
  LIMITS,
  newMessageId,
  parseMedsimMessage,
  type ChartEvent,
  type FacultyMarks,
  type MarkDomain,
  type MedsimMessage,
} from './messages';

const LOG = '[medsim:observer]';

export interface ObserverJoin {
  appId: string;
  channel: string;
  uid: number;
  token: string;
  role: string;
  caseId: string;
  studentUid: number;
  linkUid: number;
  agentUid: number;
  expiresAt: number;
}

export interface SentNote {
  id: string;
  text: string;
  at: number;
  delivered: boolean;
}

export interface ObserverView {
  phase: 'idle' | 'joining' | 'live' | 'ended' | 'error';
  error: string;
  caseId: string;
  caseLabel: string;
  osce: boolean;
  startedAt: number;
  expiresAt: number;
  lines: TranscriptLine[];
  events: ChartEvent[];
  studentInCall: boolean;
  patientInCall: boolean;
  /** The student's MedSim tab is connected (it relays whispers and marks). */
  studentOnline: boolean;
  /** Snapshot (case label, earlier transcript) received. */
  synced: boolean;
  whispers: SentNote[];
  marks: { id: string; sentAt: number; delivered: boolean } | null;
  /** 'audience' = live profile audience role; 'listen-only' = fallback. */
  audioMode: 'audience' | 'listen-only' | '';
}

const INITIAL: ObserverView = {
  phase: 'idle',
  error: '',
  caseId: '',
  caseLabel: '',
  osce: false,
  startedAt: 0,
  expiresAt: 0,
  lines: [],
  events: [],
  studentInCall: false,
  patientInCall: false,
  studentOnline: false,
  synced: false,
  whispers: [],
  marks: null,
  audioMode: '',
};

function friendlyJoinError(status: number, detail: string): string {
  if (status === 404) return 'This observer link is invalid or has expired. Ask the student for a new one.';
  if (status === 429) return 'This observer link has been used too many times. Ask the student for a new one.';
  if (status === 401) return 'Open the link on the same MedSim site the student is using.';
  if (status === 503) return 'Voice is not configured on this MedSim server.';
  return detail || `Could not join (${status}).`;
}

export class ObserverSession {
  private view: ObserverView = INITIAL;
  private listeners = new Set<() => void>();
  private audioCtx: AudioContext | null = null;
  private studentPlayer: PcmPlayer | null = null;
  private agentPlayer: PcmPlayer | null = null;
  private ws: WebSocket | null = null;
  private assembler = new TranscriptAssembler();
  private join: ObserverJoin | null = null;
  private helloTimer: number | null = null;
  private fallbackTimer: number | null = null;
  private examiner = '';
  private closed = false;

  getView = (): ObserverView => this.view;

  subscribe = (l: () => void): (() => void) => {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  };

  private set(patch: Partial<ObserverView>) {
    this.view = { ...this.view, ...patch };
    this.listeners.forEach((l) => l());
  }

  /** 0..1 speaking level for the student or the AI patient. */
  level(who: 'student' | 'patient'): number {
    if (who === 'student') {
      return this.studentPlayer?.getOutputLevel() ?? 0;
    }
    return this.agentPlayer?.getOutputLevel() ?? 0;
  }

  /** Must run inside a click (autoplay policy for audio). */
  async start(code: string, examiner: string): Promise<void> {
    if (this.view.phase === 'joining' || this.view.phase === 'live') return;
    this.examiner = examiner.trim().slice(0, LIMITS.examiner);
    this.closed = false;
    this.set({ ...INITIAL, phase: 'joining' });

    try {
      const r = await fetch('/voice/observer/join', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
      });
      if (!r.ok) {
        let detail = '';
        try { detail = ((await r.json()) as { detail?: string }).detail ?? ''; } catch { /* non-JSON */ }
        throw new Error(friendlyJoinError(r.status, detail));
      }
      const j = (await r.json()) as ObserverJoin;
      this.join = j;
      this.set({ caseId: j.caseId, caseLabel: j.caseId, expiresAt: j.expiresAt, audioMode: 'audience' });
      if (this.closed) return;

      if (!this.audioCtx) {
        const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        this.audioCtx = new AudioCtx();
      }
      if (this.audioCtx.state === 'suspended') {
        try { await this.audioCtx.resume(); } catch { /* noop */ }
      }

      this.studentPlayer = new PcmPlayer(this.audioCtx, 16000);
      this.agentPlayer = new PcmPlayer(this.audioCtx, 24000);

      await this.startWs(j);
      if (this.closed) return;
      this.set({ phase: 'live' });
    } catch (err) {
      console.warn(LOG, 'join failed:', err);
      this.teardown();
      this.set({ phase: 'error', error: err instanceof Error ? err.message : String(err) });
    }
  }

  private async startWs(j: ObserverJoin): Promise<void> {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const url = `${proto}//${host}/voice/observer/ws?channel=${encodeURIComponent(j.channel)}&uid=${encodeURIComponent(String(j.uid))}&token=${encodeURIComponent(j.token)}`;

    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    this.ws = ws;

    await new Promise<void>((resolve, reject) => {
      let settled = false;
      const timer = window.setTimeout(() => {
        if (!settled) {
          settled = true;
          reject(new Error('Observer WebSocket connection timed out.'));
        }
      }, 10000);

      ws.onopen = () => {
        if (!settled) {
          settled = true;
          window.clearTimeout(timer);
          resolve();
        }
      };

      ws.onerror = () => {
        if (!settled) {
          settled = true;
          window.clearTimeout(timer);
          reject(new Error('Observer WebSocket connection failed.'));
        }
      };
    });

    if (this.closed) {
      try { ws.close(); } catch { /* noop */ }
      return;
    }

    const self = String(j.uid);
    const link = String(j.linkUid);

    ws.onmessage = (ev: MessageEvent) => {
      if (ev.data instanceof ArrayBuffer) {
        if (ev.data.byteLength < 2) return;
        const view = new Uint8Array(ev.data);
        const prefix = view[0];
        const pcm = ev.data.slice(1);
        if (prefix === 0x01) {
          // 16kHz student PCM
          this.studentPlayer?.enqueue(pcm);
          if (!this.view.studentInCall) this.set({ studentInCall: true });
        } else if (prefix === 0x02) {
          // 24kHz agent PCM
          this.agentPlayer?.enqueue(pcm);
          if (!this.view.patientInCall) this.set({ patientInCall: true });
        }
        return;
      }

      if (typeof ev.data === 'string') {
        try {
          const obj = JSON.parse(ev.data);
          if (!obj || typeof obj !== 'object') return;

          // Agent transcripts or state
          const agent = parseAgentMessage(obj);
          if (agent) {
            if (this.assembler.apply(agent)) this.set({ lines: this.assembler.lines() });
            return;
          }

          // Presence events
          if (obj.eventType === 'SNAPSHOT') {
            const list = obj.snapshot ?? [];
            const online = list.some((u: any) => String(u.userId || u.uid) === link);
            this.set({ studentOnline: online });
            return;
          }
          if (obj.eventType === 'REMOTE_JOIN') {
            const who = String(obj.publisher || obj.userId || obj.from || '');
            if (who === link) {
              this.set({ studentOnline: true });
              this.sayHello();
            }
            return;
          }
          if (obj.eventType === 'REMOTE_LEAVE' || obj.eventType === 'REMOTE_TIMEOUT') {
            const who = String(obj.publisher || obj.userId || obj.from || '');
            if (who === link) {
              this.set({ studentOnline: false });
            }
            return;
          }

          // Medsim message
          const from = String(obj.publisher || obj.from || obj.userId || '');
          if (from === self) return;
          const m = parseMedsimMessage(obj);
          if (m && isForMe(m, self, 'USER')) {
            this.onMedsim(m);
          }
        } catch {
          /* noop */
        }
      }
    };

    ws.onclose = () => {
      if (!this.closed && this.view.phase === 'live') {
        this.set({ phase: 'ended', error: 'The observer session ended.' });
      }
    };

    let tries = 0;
    this.sayHello();
    this.helloTimer = window.setInterval(() => {
      if (this.view.synced || ++tries > 6) {
        if (this.helloTimer !== null) window.clearInterval(this.helloTimer);
        this.helloTimer = null;
        return;
      }
      this.sayHello();
    }, 5000);
  }

  private sayHello() {
    void this.sendToStudent({ object: 'medsim.hello', examiner: this.examiner });
  }

  private onMedsim(m: MedsimMessage) {
    switch (m.object) {
      case 'medsim.event':
        this.set({ events: [...this.view.events, ...m.events].slice(-300) });
        break;
      case 'medsim.snapshot': {
        if (this.view.synced) break;
        const seeded = this.assembler.seed(m.transcript);
        this.set({
          synced: true,
          studentOnline: true,
          caseLabel: m.caseLabel || this.view.caseLabel,
          osce: m.osce,
          startedAt: m.startedAt,
          events: [...m.events, ...this.view.events.filter((e) => !m.events.some((s) => s.label === e.label && s.at === e.at))],
          ...(seeded ? { lines: this.assembler.lines() } : {}),
        });
        break;
      }
      case 'medsim.ack':
        if (m.what === 'whisper') {
          this.set({ whispers: this.view.whispers.map((w) => (w.id === m.id ? { ...w, delivered: true } : w)) });
        } else if (this.view.marks?.id === m.id) {
          this.set({ marks: { ...this.view.marks, delivered: true } });
        }
        break;
      default:
        break;
    }
  }

  private async sendToStudent(m: MedsimMessage): Promise<boolean> {
    const j = this.join;
    const ws = this.ws;
    if (!j || !ws || ws.readyState !== WebSocket.OPEN) return false;
    const to = String(j.linkUid);
    try {
      ws.send(encodeMedsim({ ...m, to } as unknown as MedsimMessage));
      return true;
    } catch (err) {
      console.warn(LOG, 'sendToStudent failed:', err);
      return false;
    }
  }

  async sendWhisper(text: string): Promise<boolean> {
    const clean = text.replace(/\s+/g, ' ').trim().slice(0, LIMITS.whisper);
    if (!clean) return false;
    const note: SentNote = { id: newMessageId(), text: clean, at: Date.now(), delivered: false };
    this.set({ whispers: [...this.view.whispers, note].slice(-20) });
    const ok = await this.sendToStudent({ object: 'medsim.whisper', id: note.id, text: clean, examiner: this.examiner, at: note.at });
    if (!ok) this.set({ whispers: this.view.whispers.filter((w) => w.id !== note.id) });
    return ok;
  }

  async sendMarks(scores: Record<MarkDomain, number>, comment: string): Promise<boolean> {
    const marks: FacultyMarks = {
      id: this.view.marks?.id ?? newMessageId(),
      caseId: this.view.caseId,
      scores,
      comment: comment.trim().slice(0, LIMITS.comment),
      examiner: this.examiner,
      sentAt: Date.now(),
    };
    this.set({ marks: { id: marks.id, sentAt: marks.sentAt, delivered: false } });
    return this.sendToStudent({ object: 'medsim.marks', marks });
  }

  private teardown() {
    if (this.helloTimer !== null) window.clearInterval(this.helloTimer);
    this.helloTimer = null;
    if (this.fallbackTimer !== null) window.clearTimeout(this.fallbackTimer);
    this.fallbackTimer = null;
    if (this.studentPlayer) {
      this.studentPlayer.close();
      this.studentPlayer = null;
    }
    if (this.agentPlayer) {
      this.agentPlayer.close();
      this.agentPlayer = null;
    }
    if (this.ws) {
      const ws = this.ws;
      this.ws = null;
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      try { ws.close(); } catch { /* noop */ }
    }
  }

  leave() {
    this.closed = true;
    this.teardown();
    if (this.view.phase !== 'idle') this.set({ phase: 'ended' });
  }
}
