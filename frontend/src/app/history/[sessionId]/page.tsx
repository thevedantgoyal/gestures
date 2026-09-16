"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

import { SIGN_LABELS } from "../../../lib/signVocab";

const API_BASE = "http://localhost:8000";

const GESTURE_LABELS: Record<string, string> = {
  exit_pointing: "Exit Pointing",
  seatbelt_demo: "Seatbelt Demo",
  ...SIGN_LABELS,
};

type Attempt = {
  id: number;
  session_id: string;
  mode: string;
  gesture: string;
  correct: boolean | null;
  score: number;
  coaching_text: string | null;
  meaning: string | null;
  timestamp: string | null;
};

type TimelineEvent = {
  t_ms: number;
  gesture: string;
  correct: boolean | null;
  score: number;
  coaching_text?: string | null;
};

type SessionRecording = {
  session_id: string;
  mode: string;
  content_type: string;
  duration_ms: number | null;
  timeline: TimelineEvent[];
  video_url: string;
  created_at: string | null;
};

type SessionDetail = {
  session_id: string;
  mode: string;
  started_at: string | null;
  total_attempts: number;
  summary: { text: string };
  attempts: Attempt[];
  recording?: SessionRecording | null;
};

function formatWhen(iso: string | null | undefined) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "medium",
  });
}

function modeLabel(mode: string) {
  return mode === "sign_language" ? "Sign Language" : "Aviation Training";
}

function gestureLabel(gesture: string) {
  return GESTURE_LABELS[gesture] ?? gesture;
}

function formatMs(ms: number) {
  const total = Math.max(0, Math.round(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function AttemptSkeleton() {
  return (
    <div className="flex flex-col gap-3" aria-hidden>
      {[0, 1, 2, 3].map((key) => (
        <div
          key={key}
          className="animate-pulse border border-ink/10 bg-surface px-5 py-4"
        >
          <div className="flex justify-between gap-3">
            <div className="h-5 w-36 rounded bg-ink/10" />
            <div className="h-4 w-10 rounded bg-ink/10" />
          </div>
          <div className="mt-3 h-3 w-48 rounded bg-ink/10" />
          <div className="mt-4 h-8 w-24 rounded bg-ink/10" />
        </div>
      ))}
    </div>
  );
}

export default function SessionDetailPage() {
  const params = useParams<{ sessionId: string }>();
  const sessionId = params.sessionId;
  const videoRef = useRef<HTMLVideoElement>(null);

  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [playMs, setPlayMs] = useState(0);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;

    setLoading(true);
    fetch(`${API_BASE}/sessions/${sessionId}`)
      .then(async (response) => {
        if (response.status === 503) {
          throw new Error(
            "Can't reach the database right now — check that Postgres is running",
          );
        }
        if (!response.ok) {
          throw new Error(
            response.status === 404
              ? "Session not found"
              : `Could not load session (${response.status})`,
          );
        }
        return response.json() as Promise<SessionDetail>;
      })
      .then((data) => {
        if (!cancelled) {
          setDetail(data);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load session");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const timeline = detail?.recording?.timeline ?? [];
  const activeEvent = useMemo(() => {
    if (!timeline.length) return null;
    let current: TimelineEvent | null = null;
    for (const event of timeline) {
      if (event.t_ms <= playMs) current = event;
      else break;
    }
    return current;
  }, [timeline, playMs]);

  const videoSrc = detail?.recording?.video_url
    ? `${API_BASE}${detail.recording.video_url}`
    : null;

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-8 px-6 pb-20 pt-10">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs text-muted">
            <Link href="/history" className="text-accent hover:underline">
              History
            </Link>
            <span className="mx-2 text-ink/30">/</span>
            Session
          </p>
          <h1 className="font-heading mt-1 text-3xl font-medium tracking-tight text-ink">
            {detail ? modeLabel(detail.mode) : "Session"}
          </h1>
          {detail && (
            <p className="mt-1 text-sm text-muted">
              {formatWhen(detail.started_at)} · {detail.total_attempts} attempt
              {detail.total_attempts === 1 ? "" : "s"}
              {detail.summary?.text ? ` · ${detail.summary.text}` : ""}
            </p>
          )}
        </div>
        <Link
          href="/"
          className="shrink-0 text-sm text-accent underline-offset-2 hover:underline"
        >
          Back to practice
        </Link>
      </div>

      {error && (
        <div className="border border-alert/30 bg-alert/5 px-4 py-3 text-sm text-alert">
          {error}
        </div>
      )}

      {loading && <AttemptSkeleton />}

      {!loading && detail && videoSrc && (
        <section className="border border-ink/15 bg-surface p-4">
          <p className="text-xs font-medium text-muted">Session replay</p>
          <video
            ref={videoRef}
            src={videoSrc}
            controls
            className="mt-3 aspect-video w-full bg-ink/90"
            onTimeUpdate={() => {
              const video = videoRef.current;
              if (!video) return;
              setPlayMs(video.currentTime * 1000);
            }}
          />
          {activeEvent && (
            <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
              <span className="text-muted">{formatMs(activeEvent.t_ms)}</span>
              <span className="font-medium text-ink">
                {gestureLabel(activeEvent.gesture)}
              </span>
              {activeEvent.correct !== null && (
                <span
                  className={`px-2 py-0.5 text-xs font-medium text-white ${
                    activeEvent.correct ? "bg-success" : "bg-alert"
                  }`}
                >
                  {activeEvent.correct ? "Correct" : "Incorrect"}
                </span>
              )}
              <span className="tabular-nums text-muted">
                {Math.round(Math.max(0, Math.min(1, activeEvent.score)) * 100)}%
              </span>
            </div>
          )}
          {timeline.length > 0 && (
            <ol className="mt-4 max-h-48 space-y-2 overflow-y-auto border-t border-ink/10 pt-3 text-xs">
              {timeline.map((event, index) => (
                <li key={`${event.t_ms}-${index}`}>
                  <button
                    type="button"
                    className="flex w-full items-center justify-between gap-2 text-left text-muted hover:text-ink"
                    onClick={() => {
                      const video = videoRef.current;
                      if (!video) return;
                      video.currentTime = event.t_ms / 1000;
                      setPlayMs(event.t_ms);
                    }}
                  >
                    <span>
                      {formatMs(event.t_ms)} · {gestureLabel(event.gesture)}
                    </span>
                    <span>
                      {event.correct === null
                        ? "—"
                        : event.correct
                          ? "Correct"
                          : "Incorrect"}
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          )}
        </section>
      )}

      {!loading && detail && (
        <ol className="flex flex-col gap-3">
          {detail.attempts.map((attempt) => {
            const percent = Math.round(
              Math.max(0, Math.min(1, attempt.score)) * 100,
            );
            const isAviation = detail.mode === "aviation";

            return (
              <li
                key={attempt.id}
                className="border border-ink/15 bg-surface px-5 py-4"
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="font-heading text-lg font-medium text-ink">
                      {gestureLabel(attempt.gesture)}
                    </p>
                    <p className="mt-1 text-xs text-muted">
                      {formatWhen(attempt.timestamp)}
                    </p>
                  </div>
                  <span className="text-sm font-medium text-ink tabular-nums">
                    {percent}%
                  </span>
                </div>

                {isAviation ? (
                  <div className="mt-3">
                    {attempt.correct !== null && (
                      <span
                        className={`inline-block px-3 py-1 text-xs font-medium text-white ${
                          attempt.correct ? "bg-success" : "bg-alert"
                        }`}
                      >
                        {attempt.correct ? "Correct" : "Incorrect"}
                      </span>
                    )}
                    {attempt.coaching_text && (
                      <div className="mt-3 border border-accent/30 bg-accent/5 px-3 py-2">
                        <p className="text-xs font-medium text-accent">
                          Coaching
                        </p>
                        <p className="mt-1 text-sm leading-relaxed text-ink">
                          {attempt.coaching_text}
                        </p>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="mt-3 font-heading text-2xl font-medium text-ink">
                    {attempt.meaning ?? "—"}
                  </p>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </main>
  );
}
