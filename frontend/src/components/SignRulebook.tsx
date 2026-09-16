"use client";

import { SIGN_VOCAB, type SignGesture } from "../lib/signVocab";

type SignRulebookProps = {
  open: boolean;
  selectedGesture: SignGesture;
  referenceStatus: Record<string, "recorded" | "default"> | null;
  recordBusy: boolean;
  onClose: () => void;
  onSelect: (gesture: SignGesture) => void;
  onRecord: (gesture: SignGesture) => void;
  onResetOne: (gesture: SignGesture) => void;
  onResetAll: () => void;
};

export function SignRulebook({
  open,
  selectedGesture,
  referenceStatus,
  recordBusy,
  onClose,
  onSelect,
  onRecord,
  onResetOne,
  onResetAll,
}: SignRulebookProps) {
  if (!open) {
    return null;
  }

  const selected =
    SIGN_VOCAB.find((item) => item.key === selectedGesture) ?? SIGN_VOCAB[0];
  if (!selected) {
    return null;
  }

  const recorded = referenceStatus?.[selected.key] === "recorded";

  return (
    <aside
      className="absolute left-0 top-0 z-20 flex max-h-[calc(100vh-8rem)] w-[min(100%,20rem)] flex-col overflow-y-auto border border-l-0 border-ink/15 bg-surface shadow-sm"
      aria-label="Sign rulebook"
    >
      <div className="flex shrink-0 items-start justify-between gap-3 border-b border-ink/10 px-5 py-4">
        <div>
          <h2 className="font-heading text-lg font-medium text-ink">
            Signs you can use
          </h2>
          <p className="mt-1 text-xs leading-relaxed text-muted">
            Tap a sign to see how to do it. Record your own hand if the default
            does not match you.
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="shrink-0 text-xs text-muted hover:text-ink"
        >
          Close
        </button>
      </div>

      <ul className="grid shrink-0 grid-cols-2 gap-2 px-5 py-4">
        {SIGN_VOCAB.map((item) => {
          const isSelected = item.key === selected.key;
          const isRecorded = referenceStatus?.[item.key] === "recorded";
          return (
            <li key={item.key}>
              <button
                type="button"
                onClick={() => onSelect(item.key)}
                className={`min-h-14 w-full border px-3 py-2 text-left transition-colors ${
                  isSelected
                    ? "border-accent bg-accent/10"
                    : "border-ink/10 bg-background hover:border-ink/30"
                }`}
              >
                <span className="block text-sm font-medium text-ink">
                  {item.meaning}
                </span>
                <span className="mt-0.5 block text-xs text-muted">
                  {isRecorded ? "Recorded" : "Default"}
                </span>
              </button>
            </li>
          );
        })}
      </ul>

      <div className="shrink-0 border-t border-ink/10 px-5 py-4">
        <p className="text-xs font-medium uppercase tracking-wide text-muted">
          How to do it
        </p>
        <p className="font-heading mt-2 text-2xl font-medium text-ink">
          {selected.meaning}
        </p>
        <p className="mt-1 text-sm text-muted">{selected.label}</p>
        <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm leading-relaxed text-ink">
          {selected.steps.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
        <p
          className={`mt-4 text-xs ${
            recorded ? "text-success" : "text-muted"
          }`}
        >
          {recorded
            ? "Your recorded reference is in use for this sign."
            : "Using the built-in default. Record your hand to match you."}
        </p>
        <button
          type="button"
          onClick={() => onRecord(selected.key)}
          disabled={recordBusy}
          className="mt-3 min-h-11 w-full bg-accent px-4 py-2.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
        >
          Record this sign
        </button>
        {recorded && (
          <button
            type="button"
            onClick={() => onResetOne(selected.key)}
            disabled={recordBusy}
            className="mt-2 min-h-11 w-full border border-ink/20 px-4 py-2.5 text-sm font-medium text-ink disabled:cursor-not-allowed disabled:opacity-40"
          >
            Use default for this sign
          </button>
        )}
        <button
          type="button"
          onClick={onResetAll}
          disabled={recordBusy}
          className="mt-2 min-h-11 w-full border border-ink/20 px-4 py-2.5 text-sm font-medium text-ink disabled:cursor-not-allowed disabled:opacity-40"
        >
          Clear all recorded references
        </button>
      </div>
    </aside>
  );
}
