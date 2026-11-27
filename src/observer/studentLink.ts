/**
 * Student side of the live faculty observer.
 *
 * `inviteObserver()` asks the backend for an invite code on the live voice
 * channel and connects a separate native WebSocket identity (the "link" uid)
 * to /voice/observer/ws?channel={channel}&uid={uid}&token={token}.
 *
 * The link, not the voice `Conversation`, owns observer traffic so
 * whispers and marks still arrive after the call ends (OSCE marks are
 * usually given at the bell). It:
 *   - publishes `medsim.event` chart events (store diff, batched) to the channel,
 *   - answers `medsim.hello` with a `medsim.snapshot`,
 *   - receives `medsim.whisper` / `medsim.marks`, shows them, acks them,
 *   - forwards live 16kHz student PCM (prefix 0x01) and 24kHz agent PCM (prefix 0x02) to observers.
 */

import { useSyncExternalStore } from 'react';
import { POLYCLINIC_BED_INDEX, store } from '../game/store';
import type { GameState } from '../game/types';
import { getExistingConversation, getLastTranscript } from '../voice/conversationStore';
import { testById } from '../data/tests';
import { medicationById } from '../data/medications';
import { POLYCLINIC_DIAGNOSIS_LABELS } from '../data/polyclinicPatients';
import { EXAM_SYSTEMS } from '../agents/examClassifier';
import { attachFacultyMarks } from '../data/evalHistory';
import { chartSoFar, diffChart, type ChartLabels, type ChartSlice } from './chartEvents';
import { saveFacultyMarks } from './facultyMarks';
import {
  encodeMedsim,
  isForMe,
  LIMITS,
  parseMedsimMessage,
  type ChartEvent,
  type FacultyMarks,
  type MedsimMessage,
} from './messages';

const LOG = '[medsim:observer-link]';

export interface FacultyToast {
  id: string;
  kind: 'whisper' | 'marks' | 'call';
  text: string;
  examiner: string;
  at: number;
}

export interface LinkState {
  status: 'idle' | 'creating' | 'live' | 'error';
  error: string;
  code: string;
  inviteUrl: string;
  expiresAt: number;
  channel: string;
  caseId: string;
  /** Observer uids that said hello and have not left. */
  observers: string[];
  toasts: FacultyToast[];
}

const IDLE: LinkState = {
  status: 'idle',
  error: '',
  code: '',
  inviteUrl: '',
  expiresAt: 0,
  channel: '',
  caseId: '',
  observers: [],
  toasts: [],
};

let state: LinkState = IDLE;
const listeners = new Set<() => void>();

function setState(patch: Partial<LinkState>) {
  state = { ...state, ...patch };
  listeners.forEach((l) => l());
}

export function getLinkState(): LinkState {
  return state;
}

export function subscribeLink(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function useObserverLink(): LinkState {
  return useSyncExternalStore(subscribeLink, getLinkState);
}

export function dismissFacultyToast(id: string) {
  setState({ toasts: state.toasts.filter((t) => t.id !== id) });
}

// ── chart labels ────────────────────────────────────────────────

const LABELS: ChartLabels = {
  test: (id) => testById(id)?.name ?? id,
  medication: (id) => medicationById(id)?.name ?? id,
  diagnosis: (id) => POLYCLINIC_DIAGNOSIS_LABELS[id] ?? id.replace(/-/g, ' '),
  exam: (sys) => EXAM_SYSTEMS.find((s) => s.id === sys)?.label ?? sys,
};

const sliceOf = (s: GameState): ChartSlice => ({
  patient: s.polyclinic.patient,
  coachAlerts: s.coachAlerts,
  agentRuns: s.agentRuns,
});

// ── the live link ───────────────────────────────────────────────

interface InviteResponse {
  code: string;
  inviteUrl: string;
  expiresAt: number;
  link: { uid: number; token: string };
}

class Link {
  private ws: WebSocket | null = null;
  private unsubStore: (() => void) | null = null;
  private prev: ChartSlice;
  private pending: ChartEvent[] = [];
  private flushTimer: number | null = null;
  private expiryTimer: number | null = null;
  private history: ChartEvent[] = [];
  private closed = false;
  readonly selfUid: string;

  constructor(
    _appId: string,
    private channel: string,
    private caseId: string,
    private encounterAt: number,
    private creds: { uid: number; token: string },
    expiresAt: number,
  ) {
    this.selfUid = String(creds.uid);
    this.prev = sliceOf(store.getState());
    this.history = chartSoFar(this.prev, LABELS);
    this.expiryTimer = window.setTimeout(() => stopObserverLink(), Math.max(1000, expiresAt - Date.now()));
  }

  async start(): Promise<void> {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const url = `${proto}//${host}/voice/observer/ws?channel=${encodeURIComponent(this.channel)}&uid=${encodeURIComponent(this.selfUid)}&token=${encodeURIComponent(this.creds.token)}`;

    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    this.ws = ws;

    await new Promise<void>((resolve, reject) => {
      let settled = false;
      const timer = window.setTimeout(() => {
        if (!settled) {
          settled = true;
          reject(new Error('Observer link connection timed out.'));
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
          reject(new Error('Observer link connection failed.'));
        }
      };
    });

    if (this.closed) {
      try { ws.close(); } catch { /* noop */ }
      return;
    }

    ws.onmessage = (ev: MessageEvent) => {
      if (typeof ev.data !== 'string') return;
      try {
        const obj = JSON.parse(ev.data);
        if (!obj || typeof obj !== 'object') return;

        // Presence events
        if (obj.eventType === 'REMOTE_LEAVE' || obj.eventType === 'REMOTE_TIMEOUT') {
          const who = String(obj.publisher || obj.userId || obj.from || '');
          if (who && state.observers.includes(who)) {
            setState({ observers: state.observers.filter((o) => o !== who) });
          }
          return;
        }

        const from = String(obj.publisher || obj.from || obj.userId || '');
        if (from === this.selfUid) return;

        const m = parseMedsimMessage(obj);
        if (m && isForMe(m, this.selfUid, 'USER')) {
          this.onMessage(m, from);
        }
      } catch {
        /* noop */
      }
    };

    ws.onclose = () => {
      if (!this.closed) {
        this.close();
      }
    };

    this.unsubStore = store.subscribe(() => this.onStore());
  }

  sendAudio(prefix: number, pcm: ArrayBuffer): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    const tagged = new Uint8Array(pcm.byteLength + 1);
    tagged[0] = prefix;
    tagged.set(new Uint8Array(pcm), 1);
    try {
      this.ws.send(tagged.buffer);
    } catch {
      /* noop */
    }
  }

  sendTranscript(msg: Record<string, unknown>): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    try {
      this.ws.send(JSON.stringify(msg));
    } catch {
      /* noop */
    }
  }

  private onStore() {
    const next = sliceOf(store.getState());
    if (next.patient && next.patient.case.id !== this.caseId) {
      this.prev = next;
      return; // a different patient: not this observer's case
    }
    if (next.patient === this.prev.patient && next.coachAlerts === this.prev.coachAlerts && next.agentRuns === this.prev.agentRuns) return;
    const events = diffChart(this.prev, next, LABELS);
    this.prev = next;
    if (!events.length) return;
    this.history = [...this.history, ...events].slice(-200);
    this.pending.push(...events);
    if (this.flushTimer === null) this.flushTimer = window.setTimeout(() => this.flush(), 350);
  }

  private flush() {
    this.flushTimer = null;
    while (this.pending.length) {
      const batch = this.pending.splice(0, LIMITS.eventsPerMessage);
      void this.publishChannel({ object: 'medsim.event', caseId: this.caseId, events: batch });
    }
  }

  private async publishChannel(m: MedsimMessage) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    try {
      this.ws.send(encodeMedsim(m));
    } catch (err) {
      console.warn(LOG, 'channel publish failed:', err);
    }
  }

  private async publishTo(uid: string, m: MedsimMessage) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    try {
      this.ws.send(encodeMedsim({ ...m, to: uid } as unknown as MedsimMessage));
    } catch (err) {
      console.warn(LOG, 'publishTo failed:', err);
    }
  }

  private snapshot(): MedsimMessage {
    const s = store.getState();
    const p = s.polyclinic.patient?.case.id === this.caseId ? s.polyclinic.patient : s.lastEncounter;
    const c = p?.case;
    const transcript = getLastTranscript(this.caseId)
      .slice(-60)
      .map((m) => ({ who: m.role === 'user' ? ('doctor' as const) : ('patient' as const), text: m.content }));
    return {
      object: 'medsim.snapshot',
      caseId: this.caseId,
      caseLabel: c ? `${c.name}, ${c.age}${c.gender} · ${c.chiefComplaint}` : this.caseId,
      osce: !!s.osce && s.osce.caseId === this.caseId,
      startedAt: this.encounterAt,
      transcript,
      events: this.history,
    };
  }

  private onMessage(m: MedsimMessage, from: string) {
    switch (m.object) {
      case 'medsim.hello':
        if (!state.observers.includes(from)) setState({ observers: [...state.observers, from] });
        void this.publishTo(from, this.snapshot());
        break;
      case 'medsim.whisper':
        pushToast({ id: m.id, kind: 'whisper', text: m.text, examiner: m.examiner, at: Date.now() });
        void this.publishTo(from, { object: 'medsim.ack', id: m.id, what: 'whisper' });
        break;
      case 'medsim.marks': {
        const marks: FacultyMarks = { ...m.marks, caseId: this.caseId, encounterAt: this.encounterAt, receivedAt: Date.now() };
        saveFacultyMarks(marks);
        attachFacultyMarks(marks);
        pushToast({ id: marks.id, kind: 'marks', text: 'Faculty marks received', examiner: marks.examiner, at: Date.now() });
        void this.publishTo(from, { object: 'medsim.ack', id: marks.id, what: 'marks' });
        break;
      }
      default:
        break;
    }
  }

  close() {
    this.closed = true;
    if (this.flushTimer !== null) window.clearTimeout(this.flushTimer);
    if (this.expiryTimer !== null) window.clearTimeout(this.expiryTimer);
    this.unsubStore?.();
    this.unsubStore = null;
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
}

function pushToast(t: FacultyToast) {
  if (state.toasts.some((x) => x.id === t.id)) return;
  setState({ toasts: [...state.toasts, t].slice(-4) });
}

/** A message an observer asked Asha to pass on during her phone call. */
export function pushObserverMessage(id: string, from: string, text: string) {
  pushToast({ id, kind: 'call', text, examiner: from, at: Date.now() });
}

let link: Link | null = null;
let unloadHooked = false;

export function stopObserverLink() {
  link?.close();
  link = null;
  setState({ ...IDLE, toasts: state.toasts });
}

/** Forward live student 16kHz PCM audio chunk to active observer link. */
export function forwardStudentAudio(chunk: ArrayBuffer): void {
  link?.sendAudio(0x01, chunk);
}

/** Forward live agent 24kHz PCM audio chunk to active observer link. */
export function forwardAgentAudio(chunk: ArrayBuffer): void {
  link?.sendAudio(0x02, chunk);
}

/** Forward live transcript / agent JSON event frame to active observer link. */
export function forwardLiveTranscript(msg: Record<string, unknown>): void {
  link?.sendTranscript(msg);
}

/** Mint an invite for the live voice call and start the link. */
export async function inviteObserver(): Promise<void> {
  const conv = getExistingConversation(POLYCLINIC_BED_INDEX);
  const info = conv?.getSessionInfo();
  const s = store.getState();
  const patient = s.polyclinic.patient;
  if (!conv || !info || !patient) {
    setState({ status: 'error', error: 'Start voice first (press T): the observer listens to the live call.' });
    return;
  }
  stopObserverLink();
  setState({ status: 'creating', error: '' });
  try {
    const r = await fetch('/voice/observer/invite', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ channel: info.channel, caseId: conv.getCaseId() }),
    });
    if (!r.ok) {
      let detail = '';
      try { detail = ((await r.json()) as { detail?: string }).detail ?? ''; } catch { /* non-JSON */ }
      throw new Error(detail || `Invite failed (${r.status})`);
    }
    const inv = (await r.json()) as InviteResponse;
    const next = new Link(info.appId, info.channel, patient.case.id, patient.arrivedAt, inv.link, inv.expiresAt);
    link = next;
    if (!unloadHooked) {
      unloadHooked = true;
      window.addEventListener('pagehide', () => link?.close());
    }
    await next.start();
    if (link !== next) return;
    const url = `${window.location.origin}/?observe=${encodeURIComponent(inv.code)}`;
    setState({
      status: 'live',
      code: inv.code,
      inviteUrl: url,
      expiresAt: inv.expiresAt,
      channel: info.channel,
      caseId: patient.case.id,
      observers: [],
    });
  } catch (err) {
    console.warn(LOG, 'invite failed:', err);
    link?.close();
    link = null;
    setState({ status: 'error', error: err instanceof Error ? err.message : String(err) });
  }
}
