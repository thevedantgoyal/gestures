"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

const API_BASE = "http://localhost:8000";

type SessionSummary = {
  text: string;
  correct?: number;
  incorrect?: number;
  signs?: string[];
};

type SessionListItem = {
  session_id: string;
  mode: string;
  started_at: string | null;
  ended_at?: string | null;
  total_attempts: number;
  summary: SessionSummary;
};

function formatWhen(iso: string | null | undefined) {
  if (!iso) return "Unknown time";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function modeLabel(mode: string) {
  return mode === "sign_language" ? "Sign Language" : "Aviation Training";
}

function SessionSkeleton() {
  return (
    <div className="flex flex-col gap-4" aria-hidden>
      {[0, 1, 2].map((key) => (
        <div key={key} className="panel animate-pulse px-6 py-6">
          <div className="h-5 w-40 rounded bg-ink/10" />
          <div className="mt-3 h-3 w-52 rounded bg-ink/10" />
          <div className="mt-4 h-3 w-64 rounded bg-ink/10" />
        </div>
      ))}
    </div>
  );
}

export default function HistoryPage() {
  const [sessions, setSessions] = useState<SessionListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    setLoading(true);
    fetch(`${API_BASE}/sessions`)
      .then(async (response) => {
        if (response.status === 503) {
          throw new Error(
            "Can't reach the database right now — check that Postgres is running",
          );
        }
        if (!response.ok) {
          throw new Error(`Could not load sessions (${response.status})`);
        }
        return response.json() as Promise<SessionListItem[]>;
      })
      .then((data) => {
        if (!cancelled) {
          setSessions(data);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load history");
          setSessions([]);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-8 px-6 pb-20 pt-10">
      <div className="fade-up flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="soft-label">Gesture</p>
          <h1 className="font-heading mt-1 text-3xl font-medium tracking-tight text-ink sm:text-4xl">
            History
          </h1>
          <p className="mt-2 text-sm text-muted">
            Past practice sessions from Talk and training.
          </p>
        </div>
        <Link href="/" className="nav-pill">
          Back to practice
        </Link>
      </div>

      {error && (
        <div className="panel border-alert/25 bg-alert/5 px-4 py-3 text-sm text-alert">
          {error}
        </div>
      )}

      {loading && <SessionSkeleton />}

      {!loading && sessions && sessions.length === 0 && !error && (
        <div className="panel fade-up border-dashed px-6 py-14 text-center">
          <p className="font-heading text-2xl font-medium text-ink">
            No sessions yet
          </p>
          <p className="mt-2 text-sm text-muted">
            Start practicing to see your history here.
          </p>
          <Link href="/" className="btn-primary mt-6 inline-flex">
            Back to practice
          </Link>
        </div>
      )}

      {!loading && sessions && sessions.length > 0 && (
        <div className="fade-up flex flex-col gap-3">
          {sessions.map((session) => (
            <Link
              key={session.session_id}
              href={`/history/${session.session_id}`}
              className="panel block px-6 py-5 text-left transition-transform hover:-translate-y-0.5 hover:border-accent/35"
            >
              <div className="flex items-start justify-between gap-3">
                <span className="font-heading text-lg font-medium text-ink">
                  {modeLabel(session.mode)}
                </span>
                <span className="chip shrink-0 !py-1 text-xs">
                  {session.total_attempts} attempt
                  {session.total_attempts === 1 ? "" : "s"}
                </span>
              </div>
              <p className="mt-2 text-sm text-muted">
                {formatWhen(session.started_at)}
              </p>
              <p className="mt-3 text-sm leading-relaxed text-ink">
                {session.summary?.text}
              </p>
            </Link>
          ))}
        </div>
      )}
    </main>
  );
}
