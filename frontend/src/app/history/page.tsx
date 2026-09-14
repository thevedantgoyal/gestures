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
        <div
          key={key}
          className="animate-pulse border-2 border-ink/10 bg-surface px-6 py-6"
        >
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
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="font-heading text-3xl font-medium tracking-tight text-ink">
            History
          </h1>
          <p className="mt-1 text-sm text-muted">
            Past practice sessions from your gesture practice.
          </p>
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

      {loading && <SessionSkeleton />}

      {!loading && sessions && sessions.length === 0 && !error && (
        <div className="border-2 border-dashed border-ink/15 bg-surface px-6 py-12 text-center">
          <p className="font-heading text-xl font-medium text-ink">
            No sessions yet
          </p>
          <p className="mt-2 text-sm text-muted">
            No sessions yet — start practicing to see your history here
          </p>
          <Link
            href="/"
            className="mt-6 inline-block bg-accent px-5 py-2.5 text-sm font-medium text-white transition-colors hover:brightness-95"
          >
            Back to practice
          </Link>
        </div>
      )}

      {!loading && sessions && sessions.length > 0 && (
        <div className="flex flex-col gap-4">
          {sessions.map((session) => (
            <Link
              key={session.session_id}
              href={`/history/${session.session_id}`}
              className="border-2 border-ink/15 bg-surface px-6 py-6 text-left transition-colors hover:border-accent"
            >
              <div className="flex items-start justify-between gap-3">
                <span className="font-heading text-lg font-medium text-ink">
                  {modeLabel(session.mode)}
                </span>
                <span className="shrink-0 text-xs text-muted">
                  {session.total_attempts} attempt
                  {session.total_attempts === 1 ? "" : "s"}
                </span>
              </div>
              <p className="mt-2 text-sm text-muted">
                {formatWhen(session.started_at)}
              </p>
              <p className="mt-3 text-sm text-ink">{session.summary?.text}</p>
            </Link>
          ))}
        </div>
      )}
    </main>
  );
}
