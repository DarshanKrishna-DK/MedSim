import type { AvatarTrack } from '../AvatarVideo';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Conversation, type ConversationStatus, type SubtitleEvent } from '../../voice/conversation';
import { ensureAudioContext } from '../../voice/conversationStore';
import { TEACHER, TEACHER_GREETING, buildTeacherLessonPrompt, buildTeacherPersona } from '../../voice/teacherPersona';
import { estimateSpeechMs, type Lesson } from './lesson';
import { store, useStore } from '../../game/store';
import { LANG_LOCALE } from '../../game/lang';
import type { AgentEvent } from '../../agents/agentEvents';

// The classroom's teacher voice session. Reuses the patient voice stack as
// is: a Conversation (POST /voice/start with caseId 'teacher'), then
// /voice/context to ground it in the lesson and /voice/speak for each
// whiteboard step's narration. Typed questions go over WebSocket when live, or
// to POST /teach/ask when voice is off.
// When the session is tool-enabled, Dr. Iyer decides herself, mid-call, to
// draw on the board, pop a quiz, assign practice or book a revision
// (agent_toolsets.py); those events arrive through opts.onAgentEvents.

export interface ClassLine {
  id: number;
  who: 'teacher' | 'you';
  text: string;
}

const LIVE: ConversationStatus[] = ['ready', 'listening', 'thinking', 'speaking'];

let lineId = 0;

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${path} ${r.status}: ${(await r.text().catch(() => '')).slice(0, 160)}`);
  return (await r.json()) as T;
}

// ── offline narration (voice off) ─────────────────────────────────────
// One line at a time; a new line interrupts the previous one, like the live
// agent's INTERRUPT priority.
let offlineAudio: HTMLAudioElement | null = null;
let offlineSeq = 0;

export function stopOfflineNarration() {
  offlineSeq++;
  if (offlineAudio) {
    offlineAudio.pause();
    offlineAudio = null;
  }
}

async function playOffline(text: string, onDuration: (ms: number) => void): Promise<void> {
  stopOfflineNarration();
  const seq = offlineSeq;
  try {
    const r = await fetch('/voice/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, voice: 'F', caseId: 'teacher', gender: 'F', locale: LANG_LOCALE[store.getState().lang] }),
    });
    if (!r.ok || seq !== offlineSeq) return;
    const url = URL.createObjectURL(await r.blob());
    if (seq !== offlineSeq) return URL.revokeObjectURL(url);
    const audio = new Audio(url);
    offlineAudio = audio;
    audio.addEventListener('loadedmetadata', () => {
      if (Number.isFinite(audio.duration)) onDuration(audio.duration * 1000 + 250);
    });
    const done = () => {
      URL.revokeObjectURL(url);
      if (offlineAudio === audio) offlineAudio = null;
      onDuration(0);
    };
    audio.addEventListener('ended', done, { once: true });
    audio.addEventListener('error', done, { once: true });
    await audio.play().catch(done);
  } catch (err) {
    console.warn('[teacher] offline narration failed:', err);
  }
}

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();

export function useTeacherSession(opts: {
  onStudentFinal: (text: string, typed: boolean) => void;
  onStudentPartial?: () => void;
  /** UI events from the teacher's own tools (tool-enabled sessions only). */
  onAgentEvents?: (events: AgentEvent[]) => void;
  /** What her tools act on: topic, level, teacher email, case catalog. */
  agentContext?: () => Record<string, unknown>;
}) {
  const [status, setStatus] = useState<ConversationStatus>('uninitialized');
  const [progress, setProgress] = useState('');
  const [error, setError] = useState('');
  const [teacherCaption, setTeacherCaption] = useState<string>(TEACHER_GREETING);
  const [studentCaption, setStudentCaption] = useState<string>('');
  const [log, setLog] = useState<ClassLine[]>([]);
  const [micMuted, setMicMuted] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [avatarTrack, setAvatarTrack] = useState<AvatarTrack | null>(null);

  const convRef = useRef<Conversation | null>(null);
  const promptRef = useRef(buildTeacherPersona(store.getState().lang));
  const lessonRef = useRef<Lesson | null>(null);
  const localSpeechUntil = useRef(0);
  const lastSpoken = useRef({ text: '', at: 0 });
  const typedEcho = useRef<string | null>(null);
  const optsRef = useRef(opts);
  optsRef.current = opts;
  const logRef = useRef(log);
  logRef.current = log;

  const pushLine = useCallback((who: ClassLine['who'], text: string) => {
    setLog((l) => [...l.slice(-40), { id: ++lineId, who, text }]);
  }, []);

  const isLive = useCallback(() => {
    const c = convRef.current;
    return !!c && LIVE.includes(c.getStatus()) && !!c.getAgentId();
  }, []);

  const listeners = useRef({
    onStatus: (s: ConversationStatus) => setStatus(s),
    onProgress: (m: string) => setProgress(m),
    onError: (e: string) => setError(e),
    onSubtitle: (sub: SubtitleEvent) => {
      if (sub.who === 'patient') {
        if (sub.text === '…') return;
        setTeacherCaption(sub.text);
        if (!sub.partial) pushLine('teacher', sub.text);
        return;
      }
      setStudentCaption(sub.text);
      if (sub.partial) {
        optsRef.current.onStudentPartial?.();
        return;
      }
      const wasTyped = typedEcho.current !== null && norm(sub.text) === typedEcho.current;
      if (wasTyped) {
        typedEcho.current = null;
        return; // typed path already logged + routed it
      }
      pushLine('you', sub.text);
      optsRef.current.onStudentFinal(sub.text, false);
    },
    onAgentEvents: (events: AgentEvent[]) => optsRef.current.onAgentEvents?.(events),
  });

  // Leaving the lecture hall stops any offline narration.
  useEffect(() => stopOfflineNarration, []);

  const connect = useCallback(() => {
    stopOfflineNarration();
    if (convRef.current && convRef.current.getStatus() !== 'error') return;
    convRef.current?.dispose();
    let ctx: AudioContext;
    try {
      ctx = ensureAudioContext();
    } catch {
      setError('This browser has no Web Audio support.');
      return;
    }
    setError('');
    const lesson = lessonRef.current;
    const greeting = lesson
      ? `Achha, welcome back. We are on ${lesson.title}. Ask me anything about the board, or tell me a new topic.`
      : TEACHER_GREETING;
    const conv = new Conversation(ctx, listeners.current, {
      systemPrompt: promptRef.current,
      initialMessage: { role: 'assistant', content: greeting },
      voiceGender: TEACHER.gender,
      caseId: TEACHER.caseId,
      locale: LANG_LOCALE[store.getState().lang],
      agentContext: optsRef.current.agentContext?.(),
    });
    convRef.current = conv;
    conv.onAvatarTrack(setAvatarTrack);
    setStatus('loading');
    conv.init().catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const hangUp = useCallback(() => {
    setAvatarTrack(null);
    convRef.current?.dispose();
    convRef.current = null;
    setStatus('uninitialized');
    setProgress('');
    setMicMuted(false);
  }, []);

  // Language switched mid-call: rebuild the prompt and reconnect so ASR/TTS
  // locale, voice and persona all follow.
  const lang = useStore((s) => s.lang);
  const langRef = useRef(lang);
  useEffect(() => {
    if (langRef.current === lang) return;
    langRef.current = lang;
    const lesson = lessonRef.current;
    promptRef.current = lesson ? buildTeacherLessonPrompt(lesson, lang) : buildTeacherPersona(lang);
    const c = convRef.current;
    if (!c || c.getStatus() === 'uninitialized' || c.getStatus() === 'error') return;
    c.dispose();
    convRef.current = null;
    connect();
  }, [lang, connect]);

  const toggle = useCallback(() => {
    const c = convRef.current;
    if (c && c.getStatus() !== 'uninitialized' && c.getStatus() !== 'error') hangUp();
    else connect();
  }, [connect, hangUp]);

  const toggleMic = useCallback(() => {
    const c = convRef.current;
    if (!c) return;
    setMicMuted((m) => {
      c.setMicMuted(!m);
      return !m;
    });
  }, []);

  /** Ground the teacher in the lesson on the board (or back to the bare persona). */
  const setLesson = useCallback((lesson: Lesson | null) => {
    lessonRef.current = lesson;
    const lang = store.getState().lang;
    promptRef.current = lesson ? buildTeacherLessonPrompt(lesson, lang) : buildTeacherPersona(lang);
    const c = convRef.current;
    const agentId = c?.getAgentId();
    if (agentId) {
      const ctx = c?.isToolEnabled() ? optsRef.current.agentContext?.() : undefined;
      postJson('/voice/context', { agentId, systemPrompt: promptRef.current, ...(ctx ? { agentContext: ctx } : {}) }).catch((e) =>
        console.warn('[teacher] context update failed:', e),
      );
    }
  }, []);

  /** True while a live session whose agent runs its own tools is connected. */
  const toolsEnabled = useCallback(() => isLive() && !!convRef.current?.isToolEnabled(), [isLive]);

  /** Teacher says a line: spoken by the live agent, or captioned with a
   *  simulated mouth when voice is off. */
  const speak = useCallback(
    (text: string, opts?: { log?: boolean }) => {
      const clean = text.trim();
      if (!clean) return;
      // StrictMode re-runs effects; never send the same line twice in a row.
      const now = performance.now();
      if (lastSpoken.current.text === clean && now - lastSpoken.current.at < 800) return;
      lastSpoken.current = { text: clean, at: now };
      setTeacherCaption(clean);
      if (opts?.log) pushLine('teacher', clean);
      const agentId = isLive() ? convRef.current!.getAgentId() : null;
      if (agentId) {
        postJson('/voice/speak', { agentId, text: clean, priority: 'INTERRUPT' }).catch((e) => {
          console.warn('[teacher] speak failed:', e);
          localSpeechUntil.current = performance.now() + estimateSpeechMs(clean);
        });
      } else {
        // Voice off: no live call, but Dr. Iyer still speaks the lesson in her
        // Murf voice (HTTP TTS). The caption and mouth follow the real audio.
        localSpeechUntil.current = performance.now() + estimateSpeechMs(clean);
        void playOffline(clean, (ms) => {
          localSpeechUntil.current = performance.now() + ms;
        });
      }
    },
    [isLive, pushLine],
  );

  /** Typed question. Topic routing happens in the caller (onStudentFinal). */
  const ask = useCallback(
    async (text: string) => {
      const clean = text.trim();
      if (!clean) return;
      pushLine('you', clean);
      setStudentCaption(clean);
      optsRef.current.onStudentFinal(clean, true);
    },
    [pushLine],
  );

  /** Send a question to the teacher (after the caller decided it is not a topic request). */
  const sendQuestion = useCallback(
    async (text: string, typed: boolean) => {
      const c = convRef.current;
      if (isLive() && c) {
        if (typed) {
          typedEcho.current = norm(text);
          await c.sendTextMessage(text);
        }
        return; // spoken questions are already in the agent's turn
      }
      if (!typed) return;
      setThinking(true);
      try {
        const history = [...logRef.current, { id: 0, who: 'you' as const, text }]
          .slice(-12)
          .map((l) => ({ role: l.who === 'you' ? ('user' as const) : ('assistant' as const), content: l.text.slice(0, 1800) }));
        while (history.length && history[0].role === 'assistant') history.shift();
        const merged: typeof history = [];
        for (const m of history) {
          const last = merged[merged.length - 1];
          if (last && last.role === m.role) last.content += '\n' + m.content;
          else merged.push({ ...m });
        }
        const { reply } = await postJson<{ reply: string }>('/teach/ask', { system: promptRef.current, messages: merged });
        speak(reply, { log: true });
      } catch (e) {
        setError(e instanceof Error ? `Teacher reply failed: ${e.message}` : 'Teacher reply failed.');
      } finally {
        setThinking(false);
      }
    },
    [isLive, speak],
  );


  /** 0..1 mouth opening for the teacher's lip-sync. */
  const mouth = useCallback(() => {
    const c = convRef.current;
    const live = c ? c.getMouthAmplitude() : 0;
    if (live > 0) return live;
    const now = performance.now();
    if (now < localSpeechUntil.current) {
      const t = now / 1000;
      return 0.25 + 0.3 * Math.abs(Math.sin(t * 11.3)) * (0.6 + 0.4 * Math.sin(t * 3.1));
    }
    return 0;
  }, []);

  /** True while the teacher is (or should be) talking. */
  const isSpeaking = useCallback(() => {
    const c = convRef.current;
    if (c && (c.getStatus() === 'speaking' || c.getMouthAmplitude() > 0.04)) return true;
    return performance.now() < localSpeechUntil.current;
  }, []);

  useEffect(
    () => () => {
      convRef.current?.dispose();
      convRef.current = null;
    },
    [],
  );

  return {
    status,
    live: LIVE.includes(status),
    progress,
    error,
    clearError: () => setError(''),
    teacherCaption,
    studentCaption,
    log,
    micMuted,
    thinking,
    connect,
    hangUp,
    toggle,
    toggleMic,
    setLesson,
    toolsEnabled,
    speak,
    ask,
    sendQuestion,
    mouth,
    isSpeaking,
    avatarTrack,
  };
}

export type TeacherSession = ReturnType<typeof useTeacherSession>;
