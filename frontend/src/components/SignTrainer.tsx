"use client";

import {
  livePathLabel,
  percentLabel,
  sourceToLivePath,
  trainStatusLabel,
  type SignLiveTest,
  type SignTrainStats,
  type SignTrainStatus,
} from "../lib/signTrain";
import { SIGN_MEANINGS, SIGN_VOCAB, type SignGesture } from "../lib/signVocab";

type SignTrainerProps = {
  open: boolean;
  stats: SignTrainStats | null;
  selectedGesture: SignGesture;
  recordBusy: boolean;
  trainBusy: boolean;
  error: string | null;
  liveTest: SignLiveTest | null;
  onClose: () => void;
  onSelect: (gesture: SignGesture) => void;
  onAddSamples: (gesture: SignGesture) => void;
  onTrain: () => void;
  onPromote: () => void;
  onUseRules: () => void;
};

function statusTone(status: SignTrainStatus): string {
  switch (status) {
    case "collecting":
      return "text-muted";
    case "trainable":
      return "text-accent";
    case "ready":
      return "text-success";
    default: {
      const exhaustive: never = status;
      return exhaustive;
    }
  }
}

function liveLabel(gesture: string | null): string {
  if (!gesture) {
    return "none";
  }
  if (gesture in SIGN_MEANINGS) {
    return SIGN_MEANINGS[gesture as SignGesture];
  }
  return gesture;
}

export function SignTrainer({
  open,
  stats,
  selectedGesture,
  recordBusy,
  trainBusy,
  error,
  liveTest,
  onClose,
  onSelect,
  onAddSamples,
  onTrain,
  onPromote,
  onUseRules,
}: SignTrainerProps) {
  if (!open) {
    return null;
  }

  const overall = stats?.overall;
  const overallPercent = Math.round((overall?.progress ?? 0) * 100);
  const selected =
    stats?.gestures.find((row) => row.gesture === selectedGesture) ??
    stats?.gestures[0] ??
    null;
  const selectedVocab =
    SIGN_VOCAB.find((item) => item.key === selectedGesture) ?? SIGN_VOCAB[0];
  const last = stats?.last_train ?? null;
  const liveIsModel = stats?.source === "model";
  const busy = recordBusy || trainBusy;
  const testProb =
    liveTest && typeof liveTest.prob === "number" ? liveTest.prob : null;

  return (
    <aside
      className="flex w-full shrink-0 flex-col border border-ink/15 bg-surface"
      aria-label="Train sign model"
    >
      <div className="flex items-start justify-between gap-3 border-b border-ink/10 px-5 py-4">
        <div>
          <h2 className="font-heading text-lg font-medium text-ink">
            Train model
          </h2>
          <p className="mt-1 text-xs leading-relaxed text-muted">
            Add confirmed holds, train a real sklearn classifier, then promote
            it so Talk uses the model instead of the hardcoded rules. Wave still
            uses motion rules.
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

      <div className="border-b border-ink/10 px-5 py-4">
        <p className="text-xs font-medium uppercase tracking-wide text-muted">
          Overall
        </p>
        <p className="mt-2 text-sm text-ink">
          {overall ? (
            <>
              {overall.gold_holds} gold holds · live path{" "}
              {stats ? livePathLabel(sourceToLivePath(stats.source)) : "Rules"}
              {stats?.model.version ? ` · ${stats.model.version}` : ""}
            </>
          ) : (
            "Loading sample counts…"
          )}
        </p>
        <div className="mt-3 h-1.5 w-full bg-ink/10">
          <div
            className="h-full bg-accent"
            style={{ width: `${overallPercent}%` }}
          />
        </div>
        {last ? (
          <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
            <div className="border border-ink/10 px-3 py-2">
              <p className="text-xs text-muted">
                {last.accuracy_split === "train"
                  ? "Train accuracy"
                  : "Test accuracy"}
              </p>
              <p className="font-heading mt-1 text-xl text-ink">
                {percentLabel(last.val_accuracy)}
              </p>
            </div>
            <div className="border border-ink/10 px-3 py-2">
              <p className="text-xs text-muted">Fit accuracy</p>
              <p className="font-heading mt-1 text-xl text-ink">
                {percentLabel(last.train_accuracy)}
              </p>
            </div>
          </div>
        ) : (
          <p className="mt-2 text-xs text-muted">
            No trained model yet. Need {stats?.min_train_holds ?? 3}+ holds on
            at least two signs, then Train now.
          </p>
        )}
        {last?.warnings?.length ? (
          <p className="mt-2 text-xs text-alert">{last.warnings[0]}</p>
        ) : null}
      </div>

      <div className="border-b border-ink/10 px-5 py-4">
        <p className="text-xs font-medium uppercase tracking-wide text-muted">
          Live test
        </p>
        <p className="mt-2 text-sm text-ink">
          {liveTest?.available
            ? `${liveLabel(liveTest.label)} · ${percentLabel(testProb)} confidence · ${
                liveTest.agree === true
                  ? "agrees with rules"
                  : liveTest.agree === false
                    ? "disagrees with rules"
                    : "no rules compare"
              }`
            : liveIsModel
              ? "Model is live. Hold a sign to see confidence."
              : "Train and promote to test the model live. Rules still answer until then."}
        </p>
        {liveTest?.recognizer ? (
          <p className="mt-1 text-xs text-muted">
            This frame used {liveTest.recognizer}
            {liveTest.version ? ` (${liveTest.version})` : ""}.
          </p>
        ) : null}
      </div>

      <ul className="max-h-64 space-y-2 overflow-y-auto px-5 py-4">
        {(
          stats?.gestures ??
          SIGN_VOCAB.map((item) => ({
            gesture: item.key,
            meaning: item.meaning,
            type: item.key === "wave" ? ("motion" as const) : ("pose" as const),
            gold_holds: 0,
            status: "collecting" as const,
            progress: 0,
            live_path: "rules" as const,
            metrics: null,
          }))
        ).map((row) => {
          const isSelected = row.gesture === selectedGesture;
          const rowPercent = Math.round(row.progress * 100);
          return (
            <li key={row.gesture}>
              <button
                type="button"
                onClick={() => onSelect(row.gesture)}
                className={`w-full border px-3 py-2 text-left transition-colors ${
                  isSelected
                    ? "border-accent bg-accent/10"
                    : "border-ink/10 bg-background hover:border-ink/30"
                }`}
              >
                <span className="flex items-baseline justify-between gap-2">
                  <span className="text-sm font-medium text-ink">
                    {row.meaning}
                  </span>
                  <span className={`text-xs ${statusTone(row.status)}`}>
                    {trainStatusLabel(row.status)}
                  </span>
                </span>
                <span className="mt-0.5 flex justify-between gap-2 text-xs text-muted">
                  <span>
                    {row.gold_holds} {row.type === "motion" ? "clips" : "holds"}{" "}
                    · {livePathLabel(row.live_path)}
                  </span>
                  <span>
                    {row.metrics
                      ? `P ${percentLabel(row.metrics.precision)} · R ${percentLabel(row.metrics.recall)}`
                      : `${rowPercent}%`}
                  </span>
                </span>
                <span className="mt-2 block h-1 w-full bg-ink/10">
                  <span
                    className="block h-full bg-accent"
                    style={{ width: `${rowPercent}%` }}
                  />
                </span>
              </button>
            </li>
          );
        })}
      </ul>

      <div className="border-t border-ink/10 px-5 py-4">
        <p className="font-heading text-xl font-medium text-ink">
          {selectedVocab?.meaning ?? selected?.meaning ?? "Hello"}
        </p>
        <p className="mt-1 text-xs text-muted">
          {selected
            ? `${selected.gold_holds} confirmed ${
                selected.type === "motion" ? "clips" : "holds"
              }. Train uses ${stats?.min_train_holds ?? 3}+ holds per sign.`
            : "Select a sign, then add a confirmed sample."}
        </p>
        {selected?.metrics ? (
          <p className="mt-1 text-xs text-ink">
            Last test: precision {percentLabel(selected.metrics.precision)},
            recall {percentLabel(selected.metrics.recall)} (
            {selected.metrics.support} frames).
          </p>
        ) : null}
        <button
          type="button"
          onClick={() => onAddSamples(selectedGesture)}
          disabled={busy}
          className="mt-3 min-h-11 w-full bg-accent px-4 py-2.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
        >
          {selected?.type === "motion" ? "Add wave clips" : "Add samples"}
        </button>
        <button
          type="button"
          onClick={onTrain}
          disabled={busy || !stats?.can_train}
          className="mt-2 min-h-11 w-full border border-ink/20 px-4 py-2.5 text-sm font-medium text-ink disabled:cursor-not-allowed disabled:opacity-40"
        >
          {trainBusy ? "Training…" : "Train now"}
        </button>
        <button
          type="button"
          onClick={onPromote}
          disabled={busy || !stats?.can_promote}
          className="mt-2 min-h-11 w-full border border-ink/20 px-4 py-2.5 text-sm font-medium text-ink disabled:cursor-not-allowed disabled:opacity-40"
        >
          Promote model
        </button>
        <button
          type="button"
          onClick={onUseRules}
          disabled={busy || !liveIsModel}
          className="mt-2 min-h-11 w-full border border-ink/20 px-4 py-2.5 text-sm font-medium text-ink disabled:cursor-not-allowed disabled:opacity-40"
        >
          Use rules again
        </button>
        <p className="mt-2 text-xs leading-relaxed text-muted">
          Train fits a softmax classifier on your holds (learning rate 0.25,
          400 epochs, L2 0.001). Promote makes that model the live recognizer.
          If confidence is below 55%, rules fill in. Goodbye still needs a wave.
        </p>
        {stats?.train_blockers?.length ? (
          <p className="mt-2 text-xs text-alert">{stats.train_blockers[0]}</p>
        ) : null}
        {error ? <p className="mt-2 text-xs text-alert">{error}</p> : null}
      </div>
    </aside>
  );
}
