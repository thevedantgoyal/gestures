"use client";

export type ConversationTurn = {
  role: "user" | "assistant";
  content: string;
};

type SignCommunicatorProps = {
  voiceUnlocked: boolean;
  speaking: boolean;
  speakingTarget?: "user" | "assistant" | null;
  polishing?: boolean;
  replyLoading?: boolean;
  replyError?: string | null;
  replySource?: "ollama" | "fallback" | null;
  tokens: string[];
  sentence: string;
  lastSpoken: string | null;
  userSentence: string | null;
  assistantReply: string | null;
  conversation: ConversationTurn[];
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
  onRepeatReply: () => void;
  onClear: () => void;
  onNewConversation: () => void;
};

export function SignCommunicator({
  voiceUnlocked,
  speaking,
  speakingTarget = null,
  polishing = false,
  replyLoading = false,
  replyError = null,
  replySource = null,
  tokens,
  sentence,
  lastSpoken,
  userSentence,
  assistantReply,
  conversation,
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
  onRepeatReply,
  onClear,
  onNewConversation,
}: SignCommunicatorProps) {
  const displaySentence = sentence || lastSpoken;
  const busy = speaking || polishing || replyLoading;
  const canSpeak = Boolean(sentence) && voiceUnlocked && !busy;
  const canUndo = tokens.length > 0 && !busy;
  const canRepeat = Boolean(lastSpoken || userSentence) && voiceUnlocked && !busy;
  const canRepeatReply =
    Boolean(assistantReply) && voiceUnlocked && !busy;
  const canClear =
    tokens.length > 0 ||
    Boolean(lastSpoken) ||
    Boolean(userSentence) ||
    Boolean(assistantReply) ||
    conversation.length > 0;
  const canNewConversation =
    conversation.length > 0 || Boolean(userSentence) || Boolean(assistantReply);
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

  const statusLabel = polishing
    ? "Polishing…"
    : replyLoading
      ? "Thinking…"
      : speaking && speakingTarget === "assistant"
        ? "Speaking reply…"
        : speaking
          ? "Speaking…"
          : null;

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
          <p className="soft-label">
            {tokens.length > 0 ? "Your words" : "Your sentence"}
          </p>
          {statusLabel ? (
            <span className="pulse-soft text-sm font-semibold text-accent">
              {statusLabel}
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

        <div className="mt-5 grid grid-cols-2 gap-2.5 sm:grid-cols-3">
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
            onClick={onRepeatReply}
            disabled={!canRepeatReply}
            className="btn-secondary"
          >
            Say reply
          </button>
          <button
            type="button"
            onClick={onClear}
            disabled={!canClear}
            className="btn-secondary"
          >
            Clear
          </button>
          <button
            type="button"
            onClick={onNewConversation}
            disabled={!canNewConversation || busy}
            className="btn-secondary"
          >
            New chat
          </button>
        </div>
      </div>

      <div className="panel p-5 sm:p-6" aria-label="Conversation">
        <div className="flex items-center justify-between gap-3">
          <p className="soft-label">Conversation</p>
          {replySource === "fallback" && assistantReply ? (
            <span className="text-xs font-medium text-alert">
              Offline reply
            </span>
          ) : replySource === "ollama" && assistantReply ? (
            <span className="text-xs font-medium text-muted">Live reply</span>
          ) : null}
        </div>

        {!userSentence && !assistantReply && conversation.length === 0 ? (
          <p className="mt-3 text-base leading-relaxed text-muted">
            After you finish signing, your sentence and the assistant reply
            appear here.
          </p>
        ) : (
          <div className="mt-4 flex flex-col gap-4">
            {conversation.length > 0 ? (
              <ul className="flex flex-col gap-3" aria-live="polite">
                {conversation.map((turn, index) => {
                  const isLast = index === conversation.length - 1;
                  const offlineAssistant =
                    turn.role === "assistant" &&
                    isLast &&
                    replySource === "fallback";
                  return (
                    <li key={`${turn.role}-${index}-${turn.content.slice(0, 24)}`}>
                      <p className="text-xs font-semibold uppercase tracking-wide text-muted">
                        {turn.role === "user"
                          ? "You"
                          : offlineAssistant
                            ? "Assistant · offline"
                            : "Assistant"}
                      </p>
                      <p className="mt-1 text-base leading-relaxed text-ink">
                        {turn.content}
                      </p>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <>
                {userSentence ? (
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wide text-muted">
                      You
                    </p>
                    <p className="mt-1 text-base leading-relaxed text-ink">
                      {userSentence}
                    </p>
                  </div>
                ) : null}
                {assistantReply ? (
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wide text-muted">
                      {replySource === "fallback"
                        ? "Assistant · offline"
                        : "Assistant"}
                    </p>
                    <p className="mt-1 text-base leading-relaxed text-ink">
                      {assistantReply}
                    </p>
                  </div>
                ) : null}
              </>
            )}

            {replyLoading ? (
              <p className="pulse-soft text-sm font-semibold text-accent">
                Assistant is thinking…
              </p>
            ) : null}
            {replyError ? (
              <p className="text-sm text-alert" role="status">
                {replyError}
              </p>
            ) : replySource === "fallback" && assistantReply && !replyLoading ? (
              <p className="text-sm text-muted" role="status">
                Ollama was unavailable, so this is a canned offline reply — not
                a live model answer.
              </p>
            ) : null}
          </div>
        )}
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
