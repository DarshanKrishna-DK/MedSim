import { useEffect, useRef } from 'react';

export interface AvatarTrack {
  play?: (element: HTMLElement, config?: { fit?: 'cover' | 'contain' | 'fill' }) => void;
  stop?: () => void;
  attach?: (element: HTMLMediaElement) => void;
  detach?: () => void;
  mediaStreamTrack?: MediaStreamTrack;
}

/** Framed panel that displays an avatar video track or video element. */
export function AvatarVideo({
  track,
  label,
  style,
}: {
  track: AvatarTrack | MediaStreamTrack | null;
  label: string;
  style?: React.CSSProperties;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (!track) return;
    if (typeof (track as AvatarTrack).play === 'function') {
      const el = containerRef.current;
      if (el) {
        try { (track as AvatarTrack).play!(el, { fit: 'cover' }); } catch { /* noop */ }
      }
      return () => {
        try { (track as AvatarTrack).stop?.(); } catch { /* noop */ }
      };
    } else if (typeof MediaStreamTrack !== 'undefined' && track instanceof MediaStreamTrack) {
      if (videoRef.current) {
        videoRef.current.srcObject = new MediaStream([track]);
        videoRef.current.play().catch(() => {});
      }
      return () => {
        if (videoRef.current) videoRef.current.srcObject = null;
      };
    }
  }, [track]);

  return (
    <div
      role="img"
      aria-label={label}
      style={{ border: '3px solid #d9a520', background: '#061431', overflow: 'hidden', ...style }}
    >
      <div ref={containerRef} style={{ width: '100%', height: '100%' }}>
        <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
      </div>
    </div>
  );
}
