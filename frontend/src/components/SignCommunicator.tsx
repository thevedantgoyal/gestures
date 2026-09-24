"use client";

type SignCommunicatorProps = {
  voiceUnlocked: boolean;
  speaking: boolean;
  polishing?: boolean;
  tokens: string[];
  sentence: string;
  lastSpoken: string | null;
  currentMeaning: string | null;
  currentLabel: string | null;
  recognized: boolean;
  scorePercent: number;
  motionProgress: number | null;
  motionHint: string | null;
  handStyle: "motion" | "hold" | null;
  onEnableVoice: () => void;
  onSpeak: () => void;
  onUndo: () => void;
  onRepeat: () => void;
  onClear: () => void;
};

export function SignCommunicator({
  voiceUnlocked,
  speaking,
  polishing = false,
  tokens,
  sentence,
  lastSpoken,
  currentMeaning,
  currentLabel,
  recognized,
  scorePercent,
  motionProgress,
  motionHint,
  handStyle,
  onEnableVoice,
  onSpeak,
  onUndo,
  onRepeat,
  onClear,
}: SignCommunicatorProps) {
  const displaySentence = sentence || lastSpoken;
  const busy = speaking || polishing;
  const canSpeak = Boolean(sentence) && voiceUnlocked && !busy;
  const canUndo = tokens.length > 0 && !busy;
  const canRepeat = Boolean(lastSpoken) && voiceUnlocked && !busy;
  const moving = handStyle === "motion";
  const seeingTitle = recognized
    ? "Locked in"
    : moving
      ? "Reading motion"
      : currentLabel
        ? "Hold still"
        : "Ready for a sign";
  const seeingDetail =
    recognized && currentMeaning === "Clear"
      ? "Sentence cleared. Start a new one."
      : recognized && currentMeaning === "Undo"
        ? "Last word removed."
        : recognized && currentMeaning
          ? currentMeaning
          : moving
            ? "Dots follow your fingers — keep wagging for Goodbye."
            : currentLabel
              ? "Hold still to lock the word"
              : "Palm+thumb = Hello · 4 fingers = Undo · Fist = Clear";

  return (
    <section
      className="fade-up flex flex-col gap-4"
      aria-label="Sign language communicator"
    >
      {!voiceUnlocked && (
        <div className="panel border-accent/25 bg-accent-soft/60 px-5 py-5">
          <p className="soft-label text-accent-deep">One tap first</p>
          <p className="mt-2 text-base leading-relaxed text-ink">
            Allow voice so your sentence can be spoken out loud.
          </p>
          <button
            type="button"
            onClick={onEnableVoice}
            className="btn-primary mt-4"
          >
            Enable voice
          </button>
        </div>
      )}

      <div className="panel p-5 sm:p-6">
        <div className="flex items-center justify-between gap-3">
          <p className="soft-label">Your sentence</p>
          {polishing ? (
            <span className="pulse-soft text-sm font-semibold text-accent">
              Polishing…
            </span>
          ) : speaking ? (
            <span className="pulse-soft text-sm font-semibold text-accent">
              Speaking…
            </span>
          ) : null}
        </div>
        <p
          className="font-heading mt-3 min-h-[3.75rem] text-3xl font-medium leading-snug text-ink sm:text-[2.35rem]"
          aria-live="assertive"
          aria-atomic="true"
        >
          {displaySentence ?? "Show a sign to begin"}
        </p>

        {tokens.length > 0 && (
          <ul
            className="mt-4 flex flex-wrap gap-2"
            aria-label="Words in this sentence"
          >
            {tokens.map((token, index) => (
              <li
                key={`${token}-${index}`}
                className={`chip ${
                  index === tokens.length - 1 ? "ring-1 ring-accent/40" : ""
                }`}
              >
                {token}
              </li>
            ))}
          </ul>
        )}

        <div className="mt-5 grid grid-cols-2 gap-2.5 sm:grid-cols-4">
          <button
            type="button"
            onClick={onSpeak}
            disabled={!canSpeak}
            className="btn-primary"
          >
            {polishing ? "Polishing…" : "Speak"}
          </button>
          <button
            type="button"
            onClick={onUndo}
            disabled={!canUndo}
            className="btn-secondary"
          >
            Undo last
          </button>
          <button
            type="button"
            onClick={onRepeat}
            disabled={!canRepeat}
            className="btn-secondary"
          >
            Say again
          </button>
          <button
            type="button"
            onClick={onClear}
            disabled={tokens.length === 0 && !lastSpoken}
            className="btn-secondary"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="panel p-5 sm:p-6">
        <p className="soft-label">{seeingTitle}</p>
        {motionProgress !== null && (
          <div className="mt-3">
            <div className="mb-1.5 flex items-center justify-between text-xs">
              <span className="font-medium text-accent">
                {moving && currentLabel === "Wave"
                  ? "Watching the wag…"
                  : "Reading motion…"}
              </span>
              <span className="tabular-nums text-muted">
                {Math.round(motionProgress * 100)}%
              </span>
            </div>
            <div className="progress-track">
              <div
                className="progress-fill bg-accent"
                style={{ width: `${Math.round(motionProgress * 100)}%` }}
              />
            </div>
          </div>
        )}
        {motionHint && (
          <p className="mt-3 text-sm font-medium text-accent">{motionHint}</p>
        )}
        {currentLabel ? (
          <>
            <p className="mt-2 text-sm text-muted">{currentLabel}</p>
            <p
              className={`font-heading mt-1 text-2xl font-medium ${
                recognized
                  ? "text-success"
                  : moving
                    ? "text-accent"
                    : "text-ink-soft"
              }`}
            >
              {seeingDetail}
            </p>
            <div className="progress-track mt-4">
              <div
                className={`progress-fill ${
                  recognized ? "bg-success" : "bg-accent"
                }`}
                style={{ width: `${scorePercent}%` }}
              />
            </div>
          </>
        ) : (
          <p className="mt-3 text-base leading-relaxed text-muted">
            {seeingDetail}
          </p>
        )}
      </div>
    </section>
  );
}
