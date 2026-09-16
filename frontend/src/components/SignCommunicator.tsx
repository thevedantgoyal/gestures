"use client";

type SignCommunicatorProps = {
  voiceUnlocked: boolean;
  speaking: boolean;
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
  onRepeat: () => void;
  onClear: () => void;
};

export function SignCommunicator({
  voiceUnlocked,
  speaking,
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
  onRepeat,
  onClear,
}: SignCommunicatorProps) {
  const displaySentence = sentence || lastSpoken;
  const canSpeak = Boolean(sentence) && voiceUnlocked && !speaking;
  const canRepeat = Boolean(lastSpoken) && voiceUnlocked && !speaking;
  const moving = handStyle === "motion";
  const seeingTitle = recognized
    ? "Locked"
    : moving
      ? "Reading motion"
      : currentLabel
        ? "Hold still"
        : "Now seeing";
  const seeingDetail =
    recognized && currentMeaning === "Clear"
      ? "Sentence cleared. Start a new one."
      : recognized && currentMeaning
        ? currentMeaning
        : moving
          ? "Dots follow your fingers."
          : currentLabel
            ? "Hold still to lock the word"
            : "Hold a still open palm for Hello. A closed fist clears the sentence. Wag left and right twice for Goodbye.";

  return (
    <section
      className="flex flex-col gap-4"
      aria-label="Sign language communicator"
    >
      {!voiceUnlocked && (
        <div className="border border-accent/30 bg-accent/5 px-4 py-4">
          <p className="text-base text-ink">
            Tap once so the computer can speak your sentence out loud.
          </p>
          <button
            type="button"
            onClick={onEnableVoice}
            className="mt-3 min-h-12 bg-accent px-5 py-3 text-sm font-medium text-white transition-colors hover:brightness-95"
          >
            Enable voice
          </button>
        </div>
      )}

      <div className="border border-ink/15 bg-surface p-5">
        <p className="text-xs font-medium uppercase tracking-wide text-muted">
          Your sentence
        </p>
        <p
          className="font-heading mt-3 min-h-[3.5rem] text-3xl font-medium leading-snug text-ink sm:text-4xl"
          aria-live="assertive"
          aria-atomic="true"
        >
          {displaySentence ?? "Show a sign to start talking"}
        </p>
        {speaking && (
          <p className="mt-3 text-sm font-medium text-accent">Speaking…</p>
        )}

        {tokens.length > 0 && (
          <ul className="mt-4 flex flex-wrap gap-2" aria-label="Words in this sentence">
            {tokens.map((token, index) => (
              <li
                key={`${token}-${index}`}
                className="border border-accent/30 bg-accent/10 px-3 py-1.5 text-sm font-medium text-accent"
              >
                {token}
              </li>
            ))}
          </ul>
        )}

        <div className="mt-5 grid grid-cols-3 gap-2">
          <button
            type="button"
            onClick={onSpeak}
            disabled={!canSpeak}
            className="min-h-12 bg-accent px-3 py-3 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
          >
            Speak
          </button>
          <button
            type="button"
            onClick={onRepeat}
            disabled={!canRepeat}
            className="min-h-12 border border-ink/20 px-3 py-3 text-sm font-medium text-ink disabled:cursor-not-allowed disabled:opacity-40"
          >
            Say again
          </button>
          <button
            type="button"
            onClick={onClear}
            disabled={tokens.length === 0 && !lastSpoken}
            className="min-h-12 border border-ink/20 px-3 py-3 text-sm font-medium text-ink disabled:cursor-not-allowed disabled:opacity-40"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="border border-ink/15 bg-surface p-5">
        <p className="text-xs font-medium uppercase tracking-wide text-muted">
          {seeingTitle}
        </p>
        {motionProgress !== null && (
          <div className="mt-3">
            <div className="mb-1.5 flex items-center justify-between text-xs">
              <span className="text-accent">
                {moving && currentLabel === "Wave"
                  ? "Watching the wag…"
                  : "Reading motion…"}
              </span>
              <span className="tabular-nums text-muted">
                {Math.round(motionProgress * 100)}%
              </span>
            </div>
            <div className="h-1.5 w-full bg-ink/10">
              <div
                className="h-full bg-accent transition-[width] duration-200"
                style={{ width: `${Math.round(motionProgress * 100)}%` }}
              />
            </div>
          </div>
        )}
        {motionHint && <p className="mt-3 text-sm text-accent">{motionHint}</p>}
        {currentLabel ? (
          <>
            <p className="mt-2 text-sm text-muted">{currentLabel}</p>
            <p
              className={`font-heading mt-1 text-2xl font-medium ${
                recognized ? "text-success" : moving ? "text-accent" : "text-muted"
              }`}
            >
              {seeingDetail}
            </p>
            <div className="mt-3 h-2 w-full bg-ink/10">
              <div
                className={`h-full transition-[width] duration-300 ${
                  recognized ? "bg-success" : "bg-accent"
                }`}
                style={{ width: `${scorePercent}%` }}
              />
            </div>
          </>
        ) : (
          <p className="mt-2 text-base text-muted">{seeingDetail}</p>
        )}
      </div>

    </section>
  );
}
