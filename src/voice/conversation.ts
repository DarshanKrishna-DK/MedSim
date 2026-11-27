/**
 * Real-time voice conversation, backed by native WebSockets + WebAudio PCM streaming.
 * Replaces Agora RTC/RTM SDKs with pure browser standards:
 *   - 16kHz Int16 PCM capture via MicStreamer
 *   - 24kHz Int16 PCM playback via PcmPlayer
 *   - Native WebSocket connection to /voice/live/{channel}?token=...
 */

import { voiceMetrics } from './voiceMetrics';
import { streamClaude, type ChatMessage } from './claude';
import { mergeTurnText, parseAgentMessage, type AgentRtmMessage } from './transcriptParser';
import { startAgentEventPoller, type AgentEvent } from '../agents/agentEvents';
import { MicStreamer, PcmPlayer } from './pcmAudio';
import type { AvatarTrack } from '../components/AvatarVideo';
import { forwardStudentAudio, forwardAgentAudio, forwardLiveTranscript } from '../observer/studentLink';

const LOG = '[medsim:voice]';

const FALLBACK_PERSONA =
  'You are a patient speaking to a doctor. Keep replies to 1–2 short spoken sentences. ' +
  'Output spoken dialogue only, no stage directions, no asterisks, no brackets.';
const FALLBACK_INITIAL: { role: 'assistant'; content: string } = {
  role: 'assistant',
  content: 'Hi doc.',
};

const FAREWELL_LINES = [
  'Okay, thank you so much, doctor. Goodbye.',
  'Thanks, doctor. Take care.',
  'Alright... thank you. Bye now.',
  'Okay. Thanks for your help, doctor. Bye.',
];

export type ConversationStatus =
  | 'uninitialized'
  | 'loading'
  | 'ready'
  | 'listening'
  | 'thinking'
  | 'speaking'
  | 'error';

export interface SubtitleEvent {
  who: 'patient' | 'you';
  text: string;
  partial?: boolean;
}

export type PatientEmotion = 'neutral' | 'pain' | 'fear' | 'relief' | 'confused';

export interface ConversationListeners {
  onStatus?: (status: ConversationStatus, detail?: string) => void;
  onProgress?: (msg: string) => void;
  onSubtitle?: (sub: SubtitleEvent) => void;
  onError?: (err: string) => void;
  onEmotion?: (e: PatientEmotion) => void;
  /** Autonomous agents: UI events their server-side tools published (whiteboard,
   *  quiz, viva mark...). Polled only while a tool-enabled session is live. */
  onAgentEvents?: (events: AgentEvent[]) => void;
}

function detectEmotion(text: string): PatientEmotion {
  const t = text.toLowerCase();
  if (/\b(hurts?|pain(?:ful)?|ache|aching|ow|ouch|sore|burn(?:s|ing)?|sharp|throb|stab|rip(?:ping)?|tear(?:ing)?|crush(?:ing)?|killing me|agony|can'?t breathe|chest)\b/.test(t)) return 'pain';
  if (/\b(scared|afraid|frightened|terrified|worr(?:ied|y)|nervous|anxious|help me|please help|dying|die|gonna die|save me)\b/.test(t)) return 'fear';
  if (/\b(okay|i'?m okay|fine|better|good now|thanks|thank you|relieved?|easier|breathing better|no more pain)\b/.test(t)) return 'relief';
  if (/\b(i don'?t (?:know|understand)|not sure|confused|what does that mean|what do you mean|sorry what)\b/.test(t)) return 'confused';
  return 'neutral';
}

/** Sessions that are NOT a patient consultation. Their utterances must never
 *  reach the clinical agents (nurse orders, exam detection): a viva answer
 *  like "give atropine" is not an order for the patient in the clinic. */
const NON_PATIENT_SESSIONS = new Set(['teacher', 'viva-examiner', 'radiologist', 'followup-call', 'receptionist']);

/** Priya at Reception: her final transcripts and typed text go to the
 *  front-desk action agent (/agent/reception/act) instead of the nurse,
 *  unless the live agent is tool-enabled (then Priya runs the tools herself
 *  inside our custom LLM endpoint and no client trigger is needed). */
const RECEPTIONIST_SESSION = 'receptionist';

function runReceptionFromSession(utterance: string, source: 'voice' | 'typed') {
  void import('../agents/receptionAgent')
    .then((m) => {
      if (m.looksLikeReceptionRequest(utterance)) void m.runReceptionAction(utterance, source);
    })
    .catch((err) => console.warn(LOG, 'reception agent load failed:', err));
}

export interface ConversationOptions {
  /** Run the nurse/exam agents on this session's utterances. Defaults to true
   *  for patient consultations and false for teacher, viva and radiology. */
  clinicalAgents?: boolean;
  systemPrompt?: string;
  initialMessage?: { role: 'assistant'; content: string };
  /** Speaker gender ('M'|'F'), used by backend voice picker. */
  voiceGender?: 'M' | 'F';
  /** Stable case id, backend uses this for logging + voice slot pick. */
  caseId?: string;
  /** Preserved for back-compat. Ignored, voice ID is chosen server-side. */
  voice?: string;
  /** If set, persist/restore the conversation history to localStorage under this key. */
  storageKey?: string;
  /** BCP-47 locale hint for ASR/TTS (backend default if omitted). */
  locale?: string;
  /** Ask the backend for an avatar (undefined = backend default). */
  avatar?: boolean;
  /** Custom agent start endpoint (default /voice/start). The persona then
   *  lives server-side: `startBody` is posted as is, instead of the prompt. */
  startUrl?: string;
  startBody?: Record<string, unknown>;
  /** Custom stop endpoint (default /voice/stop), posted {agentId}. */
  stopUrl?: string;
  /** Autonomous agents (receptionist / teacher / viva-examiner): what their
   *  server-side tools may act on, sent as /voice/start `agentContext`. */
  agentContext?: Record<string, unknown>;
}

interface VoiceStartResponse {
  appId?: string;
  channel: string;
  uid: number;
  token: string;
  agentUid: number;
  agentId: string;
  avatarUid?: number;
  avatarError?: string;
  toolsEnabled?: boolean;
  wsUrl?: string;
}

// ── backend helpers ─────────────────────────────────────────────

class VoiceBackendError extends Error {}

function friendlyBackendError(status: number, body: string): string {
  const b = body.toLowerCase();
  if (
    status === 503 ||
    /not configured|missing .*key|gemini_api_key|no app ?id/.test(b)
  ) {
    return 'Voice backend not configured, add Gemini API keys to backend/.env.local.';
  }
  if (status === 502 || status === 504 || (status === 500 && body.trim() === '')) {
    return 'Voice backend unreachable, start backend/server.py on :8787.';
  }
  const detail = body.replace(/\s+/g, ' ').slice(0, 160);
  return `Voice backend error ${status}${detail ? `: ${detail}` : ''}`;
}

async function postVoice<T>(path: string, body: unknown, opts: { keepalive?: boolean } = {}): Promise<T> {
  let r: Response;
  try {
    r = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      keepalive: opts.keepalive,
    });
  } catch {
    throw new VoiceBackendError('Voice backend unreachable, start backend/server.py on :8787.');
  }
  if (!r.ok) {
    const text = await r.text().catch(() => '');
    throw new VoiceBackendError(friendlyBackendError(r.status, text));
  }
  const ct = r.headers.get('content-type') ?? '';
  return (ct.includes('application/json') ? await r.json() : (undefined as unknown)) as T;
}

function friendlyMediaError(err: unknown): string | null {
  const code = (err as { code?: string })?.code ?? '';
  const name = (err as { name?: string })?.name ?? '';
  const msg = (err as { message?: string })?.message ?? '';
  if (code === 'PERMISSION_DENIED' || name === 'NotAllowedError' || /permission denied/i.test(msg)) {
    return 'Microphone permission denied, allow mic access and press T to retry.';
  }
  if (code === 'DEVICE_NOT_FOUND' || name === 'NotFoundError') {
    return 'No microphone found.';
  }
  return null;
}

// ── page-unload safety net: stop any agents we started ─────────

const liveAgentIds = new Map<string, string>();
let unloadHookInstalled = false;
function installUnloadHook() {
  if (unloadHookInstalled || typeof window === 'undefined') return;
  unloadHookInstalled = true;
  window.addEventListener('pagehide', () => {
    for (const [agentId, stopUrl] of liveAgentIds) {
      try {
        const blob = new Blob([JSON.stringify({ agentId })], { type: 'application/json' });
        navigator.sendBeacon(stopUrl, blob);
      } catch { /* noop */ }
    }
    liveAgentIds.clear();
  });
}

function normWords(s: string): string {
  return s.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, '').replace(/\s+/g, ' ').trim();
}

type AgentState = 'idle' | 'listening' | 'thinking' | 'speaking' | 'silent';

export class Conversation {
  private messages: ChatMessage[] = [];
  private listeners: ConversationListeners = {};
  private status: ConversationStatus = 'uninitialized';

  private streamer: MicStreamer | null = null;
  private player: PcmPlayer | null = null;
  private ws: WebSocket | null = null;

  private avatarVideo: AvatarTrack | null = null;
  private avatarListeners = new Set<(t: AvatarTrack | null) => void>();
  private session: VoiceStartResponse | null = null;
  private disposed = false;
  private micMuted = false;

  private audioCtx: AudioContext;
  private smoothedAmp = 0;

  private systemPrompt: string;
  private basePrompt: string;
  private contextNotes: string[] = [];
  private initialMessage: { role: 'assistant'; content: string };
  private voiceGender: 'M' | 'F';
  private caseId: string;
  private clinicalAgents: boolean;
  private locale: string | undefined;
  private wantAvatar: boolean | undefined;
  private storageKey: string | null = null;
  private startUrl: string | null;
  private startBody: Record<string, unknown> | null;
  private stopUrl: string;
  private agentContext: Record<string, unknown> | null;
  private stopEvents: (() => void) | null = null;

  private messageSubscribers = new Set<(msgs: ReadonlyArray<ChatMessage>) => void>();

  // Latest agent text spoken, drives the 3D face emotion.
  private currentEmotion: PatientEmotion = 'neutral';

  // Transcript assembly. Key = `${role}:${turn_id}` → index into messages.
  private turnIndex = new Map<string, number>();
  private hasAgentStateFeed = false;
  private agentState: AgentState = 'idle';

  // Nurse prefilter debounce, a user turn can arrive as several finals.
  private nurseTimer: number | null = null;
  private nurseCandidate = '';
  /** Normalised typed messages sent over WebSocket, awaiting a possible echo. */
  private pendingTyped: string[] = [];

  constructor(
    audioCtx: AudioContext,
    listeners: ConversationListeners = {},
    options: ConversationOptions = {}
  ) {
    this.audioCtx = audioCtx;
    this.listeners = listeners;
    this.systemPrompt = options.systemPrompt ?? FALLBACK_PERSONA;
    this.basePrompt = this.systemPrompt;
    this.initialMessage = options.initialMessage ?? FALLBACK_INITIAL;
    this.voiceGender = options.voiceGender ?? 'M';
    this.caseId = options.caseId ?? 'unknown';
    this.clinicalAgents = options.clinicalAgents ?? !NON_PATIENT_SESSIONS.has(this.caseId);
    this.locale = options.locale;
    this.wantAvatar = options.avatar;
    this.storageKey = options.storageKey ?? null;
    this.startUrl = options.startUrl ?? null;
    this.startBody = options.startBody ?? null;
    this.stopUrl = options.stopUrl ?? '/voice/stop';
    this.agentContext = options.agentContext ?? null;

    const restored = this.loadMessages();
    this.messages = restored && restored.length > 0 ? restored : [this.initialMessage];
  }

  private saveMessages() {
    if (!this.storageKey) return;
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this.messages));
    } catch { /* quota or privacy-mode, silently ignore */ }
  }

  private loadMessages(): ChatMessage[] | null {
    if (!this.storageKey) return null;
    try {
      const raw = localStorage.getItem(this.storageKey);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return null;
      return parsed.filter(
        (m: any) => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string'
      );
    } catch {
      return null;
    }
  }

  setListeners(listeners: ConversationListeners) {
    this.listeners = listeners;
  }

  getStatus(): ConversationStatus {
    return this.status;
  }

  getMessages(): ReadonlyArray<ChatMessage> {
    return [...this.messages];
  }

  subscribeMessages(fn: (msgs: ReadonlyArray<ChatMessage>) => void): () => void {
    this.messageSubscribers.add(fn);
    return () => { this.messageSubscribers.delete(fn); };
  }

  private emitMessages() {
    const snap = [...this.messages];
    this.messageSubscribers.forEach((fn) => fn(snap));
  }

  /** Avatar video track if present, else null. */
  getAvatarTrack(): AvatarTrack | null {
    return this.avatarVideo;
  }

  /** Subscribe to avatar video changes. Fires immediately with the current track. */
  onAvatarTrack(fn: (t: AvatarTrack | null) => void): () => void {
    this.avatarListeners.add(fn);
    fn(this.avatarVideo);
    return () => { this.avatarListeners.delete(fn); };
  }

  private setAvatarTrack(t: AvatarTrack | null): void {
    this.avatarVideo = t;
    this.avatarListeners.forEach((fn) => fn(t));
  }

  /** Live mouth amplitude from the agent's PCM playback level.
   *  0..1, usable directly as a mouth-open factor. Cheap, safe to call
   *  every animation frame. */
  getMouthAmplitude(): number {
    const playerLevel = this.player?.getOutputLevel() ?? 0;
    this.smoothedAmp = this.smoothedAmp * 0.45 + playerLevel * 0.55;
    return this.smoothedAmp < 0.02 ? 0 : this.smoothedAmp;
  }

  getCurrentEmotion(): PatientEmotion {
    return this.currentEmotion;
  }

  // ── additions (non-breaking) used by the nurse agent ──────────

  /** Agent id for this session, or null if voice isn't live. */
  getAgentId(): string | null {
    return this.session?.agentId ?? null;
  }

  /** Voice channel identity of the live call (observer invites), or null. */
  getSessionInfo(): { appId: string; channel: string; uid: number; agentUid: number } | null {
    const s = this.session;
    return s ? { appId: s.appId ?? '', channel: s.channel, uid: s.uid, agentUid: s.agentUid } : null;
  }

  getCaseId(): string {
    return this.caseId;
  }

  /** True when the live agent decides on and runs its own tools (no client keyword triggers). */
  isToolEnabled(): boolean {
    return !!this.session?.toolsEnabled;
  }

  /** Update what a tool-enabled agent's tools act on (merged server-side). */
  async updateAgentContext(ctx: Record<string, unknown>): Promise<void> {
    this.agentContext = { ...(this.agentContext ?? {}), ...ctx };
    const agentId = this.getAgentId();
    if (!agentId || !this.isToolEnabled()) return;
    try {
      await postVoice('/voice/context', { agentId, agentContext: ctx });
    } catch (err) {
      console.warn(LOG, 'agent context update failed:', err);
    }
  }

  /** Mute/unmute the doctor's mic without changing conversation status. */
  isMicMuted(): boolean {
    return this.micMuted;
  }

  setMicMuted(muted: boolean) {
    this.micMuted = muted;
    this.streamer?.setMuted(muted);
  }

  /** Append a note to the patient persona and push the updated prompt to the live agent. */
  async appendContextNote(note: string): Promise<void> {
    const clean = note.trim();
    if (!clean) return;
    this.contextNotes.push(clean);
    this.systemPrompt =
      `${this.basePrompt}\n\nWHAT HAS HAPPENED DURING THIS VISIT (you are aware of these; mention them naturally only if relevant):\n` +
      this.contextNotes.map((n) => `- ${n}`).join('\n');
    const agentId = this.getAgentId();
    if (!agentId) return;
    try {
      await postVoice('/voice/context', { agentId, systemPrompt: this.systemPrompt });
    } catch (err) {
      console.warn(LOG, 'context update failed:', err);
    }
  }

  private setEmotion(e: PatientEmotion) {
    if (e === this.currentEmotion) return;
    this.currentEmotion = e;
    this.listeners.onEmotion?.(e);
  }

  private setStatus(s: ConversationStatus, detail?: string) {
    this.status = s;
    this.listeners.onStatus?.(s, detail);
  }

  // ── transcript handling ──────────────────────────────────────

  private upsertTurnMessage(role: 'user' | 'assistant', turnId: unknown, text: string) {
    const key = turnId === undefined || turnId === null ? null : `${role}:${String(turnId)}`;
    const idx = key ? this.turnIndex.get(key) : undefined;
    if (idx !== undefined && this.messages[idx]?.role === role) {
      const next = mergeTurnText(this.messages[idx].content, text, role);
      this.messages[idx] = { role, content: next };
      return next;
    }
    this.messages.push({ role, content: text });
    if (key) this.turnIndex.set(key, this.messages.length - 1);
    return text;
  }

  private handleUserTranscription(msg: Extract<AgentRtmMessage, { kind: 'user' }>) {
    const { text } = msg;
    if (!msg.final) {
      this.listeners.onSubtitle?.({ who: 'you', text, partial: true });
      if (this.status !== 'listening') this.setStatus('listening');
      return;
    }
    const echoAt = this.pendingTyped.indexOf(normWords(text));
    if (echoAt >= 0) {
      this.pendingTyped.splice(echoAt, 1);
      return;
    }
    const full = this.upsertTurnMessage('user', msg.turnId, text);
    this.listeners.onSubtitle?.({ who: 'you', text: full });
    this.saveMessages();
    this.emitMessages();
    if (!this.hasAgentStateFeed || this.agentState !== 'speaking') {
      this.setStatus('thinking', 'Thinking…');
    }
    this.scheduleNurseCheck(full);
  }

  private handleAgentTranscription(msg: Extract<AgentRtmMessage, { kind: 'assistant' }>) {
    const { text } = msg;
    if (msg.turnStatus === 0) {
      this.setEmotion(detectEmotion(text));
      this.listeners.onSubtitle?.({ who: 'patient', text, partial: true });
      if (this.status !== 'speaking') this.setStatus('speaking');
      return;
    }
    // END (1) or INTERRUPTED (2)
    const full = this.upsertTurnMessage('assistant', msg.turnId, text);
    this.setEmotion(detectEmotion(full));
    this.listeners.onSubtitle?.({ who: 'patient', text: full });
    this.saveMessages();
    this.emitMessages();
    if (!this.hasAgentStateFeed) {
      this.setStatus('ready');
    } else if (this.status !== 'speaking' && this.agentState === 'speaking') {
      this.setStatus('speaking');
    }
  }

  private handleAgentState(state: string) {
    this.hasAgentStateFeed = true;
    const s = state.toLowerCase() as AgentState;
    this.agentState = s;
    if (s === 'speaking') {
      if (this.status !== 'speaking') this.setStatus('speaking');
    } else if (s === 'thinking') {
      if (this.status !== 'thinking') this.setStatus('thinking', 'Thinking…');
    } else if (s === 'listening' || s === 'idle' || s === 'silent') {
      if (this.status === 'speaking' || this.status === 'thinking') this.setStatus('ready');
    }
  }

  private handleWsJsonMessage(msg: Record<string, unknown>) {
    const obj = String(msg.object || msg.type || '');
    msg.object = obj;

    // Diagnostics panel + debrief stats
    voiceMetrics.ingestRtm(this.caseId, msg);

    // Forward live transcript & event frames to studentLink observer if active
    forwardLiveTranscript(msg);

    if (obj === 'message.interrupt' || obj === 'interrupt') {
      this.player?.clear();
      if (this.status === 'speaking') this.setStatus('listening');
      return;
    }

    if (obj === 'state' || obj === 'agent_state' || msg.state) {
      const st = String(msg.state || msg.agent_state || '');
      if (st) this.handleAgentState(st);
      return;
    }

    if (obj === 'closed') {
      const reason = String(msg.reason || 'agent left');
      if (!this.disposed) {
        this.player?.clear();
        this.listeners.onError?.(`The patient voice agent left the call (${reason}). Press T to reconnect.`);
        this.setStatus('error', 'agent left');
      }
      return;
    }

    const m = parseAgentMessage(msg);
    if (!m) return;
    switch (m.kind) {
      case 'user':
        this.handleUserTranscription(m);
        break;
      case 'assistant':
        this.handleAgentTranscription(m);
        break;
      case 'interrupt':
        this.player?.clear();
        if (this.status === 'speaking') this.setStatus('listening');
        break;
      case 'error':
        console.warn(LOG, 'agent error:', m.raw);
        break;
    }
  }

  private scheduleNurseCheck(text: string) {
    this.nurseCandidate = text;
    if (this.nurseTimer !== null) window.clearTimeout(this.nurseTimer);
    this.nurseTimer = window.setTimeout(() => {
      this.nurseTimer = null;
      const utterance = this.nurseCandidate;
      this.nurseCandidate = '';
      if (this.disposed || !utterance) return;
      if (this.caseId === RECEPTIONIST_SESSION) {
        if (this.isToolEnabled()) return;
        return void runReceptionFromSession(utterance, 'voice');
      }
      if (!this.clinicalAgents) return;
      void Promise.all([import('../agents/examAgent'), import('../agents/nurseAgent'), import('../agents/coach'), import('../observer/observerCall')])
        .then(([ex, m, coach, obs]) => {
          if (obs.maybeHandleObserverCommand(utterance)) return;
          if (coach.isHintRequest(utterance)) return void coach.requestHint();
          if (m.isAddressedToNurse(utterance)) return void m.runNurseAction(utterance, 'voice');
          if (ex.maybeRunExamFromSpeech(utterance)) return;
          if (m.looksLikeOrder(utterance)) void m.runNurseAction(utterance, 'voice');
        })
        .catch((err) => console.warn(LOG, 'nurse agent load failed:', err));
    }, 700);
  }

  // ── lifecycle ────────────────────────────────────────────────

  async init() {
    if (this.status !== 'uninitialized' || this.disposed) return;
    this.setStatus('loading');
    installUnloadHook();

    try {
      this.listeners.onProgress?.('Starting voice agent…');
      const initialLine = this.initialMessage.content;
      const startUrl = this.startUrl ?? '/voice/start';
      const session = await postVoice<VoiceStartResponse>(
        startUrl,
        this.startUrl
          ? (this.startBody ?? {})
          : {
              caseId: this.caseId,
              systemPrompt: this.systemPrompt,
              initialLine,
              gender: this.voiceGender,
              ...(this.locale ? { locale: this.locale } : {}),
              ...(this.wantAvatar !== undefined ? { avatar: this.wantAvatar } : {}),
              ...(this.agentContext ? { agentContext: this.agentContext } : {}),
            },
      );
      if (!session || !session.channel) {
        throw new VoiceBackendError(`Voice backend not configured, ${startUrl} returned no session.`);
      }
      this.session = session;
      liveAgentIds.set(session.agentId, this.stopUrl);
      if (this.disposed) return this.teardown();

      if (this.audioCtx.state === 'suspended') {
        try { await this.audioCtx.resume(); } catch { /* noop */ }
      }

      // Initialize 24kHz PCM player for incoming audio
      this.player = new PcmPlayer(this.audioCtx, 24000);

      // Connect to native WebSocket: ws(s)://{host}/voice/live/{channel}?token=...
      this.listeners.onProgress?.('Connecting live voice stream…');
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      const wsUrl = session.wsUrl
        ? (session.wsUrl.startsWith('ws') ? session.wsUrl : `${proto}//${host}${session.wsUrl}`)
        : `${proto}//${host}/voice/live/${encodeURIComponent(session.channel)}?token=${encodeURIComponent(session.token || '')}`;

      const ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';
      this.ws = ws;

      await new Promise<void>((resolve, reject) => {
        let settled = false;
        const timer = window.setTimeout(() => {
          if (!settled) {
            settled = true;
            reject(new Error('Voice WebSocket connection timed out.'));
          }
        }, 12000);

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
            reject(new Error('Voice WebSocket connection failed.'));
          }
        };
      });

      if (this.disposed) return this.teardown();

      // Configure WebSocket message and close handlers
      ws.onmessage = (ev: MessageEvent) => {
        if (ev.data instanceof ArrayBuffer) {
          // Binary 24kHz PCM from agent
          this.player?.enqueue(ev.data);
          forwardAgentAudio(ev.data);
          if (this.status !== 'speaking' && this.agentState === 'speaking') {
            this.setStatus('speaking');
          }
        } else if (typeof ev.data === 'string') {
          try {
            const msg = JSON.parse(ev.data);
            if (msg && typeof msg === 'object') {
              this.handleWsJsonMessage(msg as Record<string, unknown>);
            }
          } catch {
            /* noop */
          }
        }
      };

      ws.onclose = (ev) => {
        if (this.disposed) return;
        const isAgentLeft = ev.reason && /agent left|leave|timeout|closed/i.test(ev.reason);
        if (isAgentLeft || this.status === 'speaking' || this.status === 'ready' || this.status === 'listening' || this.status === 'thinking') {
          this.listeners.onError?.('The patient voice agent left the call. Press T to reconnect.');
          this.setStatus('error', 'agent left');
        } else if (this.status !== 'error') {
          this.setStatus('uninitialized');
        }
      };

      // Start 16kHz PCM capture via MicStreamer
      this.listeners.onProgress?.('Enabling microphone…');
      const streamer = new MicStreamer(this.audioCtx);
      this.streamer = streamer;

      streamer.onChunk((chunk: ArrayBuffer) => {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(chunk);
        }
        forwardStudentAudio(chunk);
      });

      streamer.onLevel((lvl: number) => {
        if (lvl > 0.05 && this.status === 'ready') {
          this.setStatus('listening');
        }
      });

      await streamer.start();
      if (this.disposed) return this.teardown();

      if (this.micMuted) streamer.setMuted(true);

      this.listeners.onProgress?.('Live.');
      this.setStatus('ready');

      if (session.toolsEnabled && this.listeners.onAgentEvents) {
        this.stopEvents = startAgentEventPoller(session.channel, (evs) => {
          if (!this.disposed) this.listeners.onAgentEvents?.(evs);
        });
      }

      this.listeners.onSubtitle?.({ who: 'patient', text: '…' });
      this.setEmotion(detectEmotion(initialLine));
    } catch (err: any) {
      const msg =
        err instanceof VoiceBackendError
          ? err.message
          : friendlyMediaError(err) ?? (err?.message ?? String(err));
      console.error(LOG, 'init failed:', err);
      this.teardown();
      if (this.disposed) return;
      this.setStatus('error', msg);
      this.listeners.onError?.(msg);
      throw new Error(msg);
    }
  }

  async startListening() {
    if (this.status === 'uninitialized') return;
    this.micMuted = false;
    this.streamer?.setMuted(false);
  }

  async stopListeningAndRespond() {
    /* no-op */
  }

  cancel() {
    this.micMuted = true;
    this.streamer?.setMuted(true);
    this.player?.clear();
    this.setStatus('ready');
  }

  async sayFarewell(): Promise<void> {
    const line = FAREWELL_LINES[Math.floor(Math.random() * FAREWELL_LINES.length)];
    this.setStatus('thinking', 'Saying goodbye…');

    this.micMuted = true;
    this.streamer?.setMuted(true);

    const agentId = this.getAgentId();
    if (agentId && this.ws && this.ws.readyState === WebSocket.OPEN) {
      let spoke = false;
      try {
        await postVoice('/voice/speak', { agentId, text: line, priority: 'INTERRUPT' });
        spoke = true;
      } catch (err) {
        console.warn(LOG, 'farewell speak via POST failed, trying WS:', err);
        try {
          this.ws.send(JSON.stringify({ type: 'text', text: line, priority: 'INTERRUPT' }));
          spoke = true;
        } catch { /* noop */ }
      }
      this.setEmotion(detectEmotion(line));
      this.listeners.onSubtitle?.({ who: 'patient', text: line });

      if (spoke) {
        const deadline = Date.now() + 7000;
        let sawSpeaking = false;
        let quietSince = 0;
        while (Date.now() < deadline) {
          const audible = this.getMouthAmplitude() > 0.04;
          const speakingNow = this.status === 'speaking' || audible;
          if (speakingNow) { sawSpeaking = true; quietSince = 0; }
          else if (sawSpeaking) {
            if (!quietSince) quietSince = Date.now();
            if (Date.now() - quietSince > 700) break;
          }
          await new Promise((r) => window.setTimeout(r, 100));
        }
        await new Promise((r) => window.setTimeout(r, 300));
      }
    } else {
      this.setEmotion(detectEmotion(line));
      this.listeners.onSubtitle?.({ who: 'patient', text: line });
      this.messages.push({ role: 'assistant', content: line });
      this.emitMessages();
    }

    if (this.storageKey) {
      try { localStorage.removeItem(this.storageKey); } catch { /* noop */ }
    }
    this.setStatus('ready');
  }

  async sendTextMessage(text: string, opts?: { speak?: boolean }): Promise<void> {
    const clean = text.trim();
    if (!clean) return;
    if (this.status === 'loading') return;

    this.listeners.onSubtitle?.({ who: 'you', text: clean });
    this.messages.push({ role: 'user', content: clean });
    this.saveMessages();
    this.emitMessages();
    this.setStatus('thinking', 'Thinking…');

    const viaLiveAgent = !!(this.ws && this.ws.readyState === WebSocket.OPEN && opts?.speak !== false);
    if (this.caseId === RECEPTIONIST_SESSION && !(viaLiveAgent && this.isToolEnabled())) {
      runReceptionFromSession(clean, 'typed');
    }
    if (this.clinicalAgents) {
      void Promise.all([import('../agents/examAgent'), import('../agents/nurseAgent'), import('../agents/coach'), import('../observer/observerCall')])
        .then(([ex, m, coach, obs]) => {
          if (obs.maybeHandleObserverCommand(clean)) return;
          if (coach.isHintRequest(clean)) return void coach.requestHint();
          if (m.isAddressedToNurse(clean)) return void m.runNurseAction(clean, 'typed');
          if (ex.maybeRunExamFromSpeech(clean)) return;
          if (m.looksLikeOrder(clean)) void m.runNurseAction(clean, 'typed');
        })
        .catch(() => undefined);
    }

    if (viaLiveAgent) {
      this.pendingTyped.push(normWords(clean));
      try {
        this.ws?.send(
          JSON.stringify({
            type: 'text',
            text: clean,
            priority: 'interrupted',
            interruptable: true,
            message: clean,
          })
        );
        return;
      } catch (err) {
        console.warn(LOG, 'WS text send failed, falling back to HTTP:', err);
        this.pendingTyped.pop();
      }
    }

    const controller = new AbortController();
    let assistantText = '';
    try {
      for await (const token of streamClaude(this.systemPrompt, this.messages, controller.signal)) {
        assistantText += token;
      }
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        console.error(LOG, 'text-chat LLM error:', err);
        this.listeners.onError?.(`Patient reply failed: ${err?.message ?? err}`);
        this.setStatus('ready');
        return;
      }
    }

    const trimmed = assistantText.trim();
    if (trimmed) {
      this.setEmotion(detectEmotion(trimmed));
      this.listeners.onSubtitle?.({ who: 'patient', text: trimmed });
      this.messages.push({ role: 'assistant', content: trimmed });
      this.saveMessages();
      this.emitMessages();
    }
    this.setStatus('ready');
  }

  reset() {
    this.cancel();
    this.messages = [this.initialMessage];
    this.turnIndex.clear();
    this.emitMessages();
    if (this.storageKey) {
      try { localStorage.removeItem(this.storageKey); } catch { /* noop */ }
    }
  }

  private teardown() {
    if (this.stopEvents) {
      this.stopEvents();
      this.stopEvents = null;
    }
    if (this.nurseTimer !== null) {
      window.clearTimeout(this.nurseTimer);
      this.nurseTimer = null;
    }
    if (this.streamer) {
      this.streamer.stop();
      this.streamer = null;
    }
    if (this.player) {
      this.player.close();
      this.player = null;
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
    if (this.avatarVideo) {
      try { this.avatarVideo.stop?.(); } catch { /* noop */ }
      this.setAvatarTrack(null);
    }
    if (this.session) {
      const { agentId } = this.session;
      this.session = null;
      liveAgentIds.delete(agentId);
      postVoice(this.stopUrl, { agentId }, { keepalive: true }).catch(() => undefined);
    }
    this.smoothedAmp = 0;
  }

  dispose() {
    this.disposed = true;
    this.teardown();
    this.setStatus('uninitialized');
  }
}
