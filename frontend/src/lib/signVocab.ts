export const SIGN_VOCAB = [
  {
    key: "open_palm",
    label: "Open Palm",
    how: "Hold palm still",
    meaning: "Hello",
    steps: [
      "Face the camera with one hand in the frame.",
      "Open all five fingers — thumb out. Palm faces the camera.",
      "Hold still. Do not wag or swipe. Four fingers with thumb in is Undo, not Hello.",
    ],
  },
  {
    key: "thumbs_up",
    label: "Thumbs Up",
    how: "Thumbs up",
    meaning: "Yes",
    steps: [
      "Close the fingers into a fist.",
      "Stick the thumb straight up, clearly above the fist.",
      "A fully closed hand (thumb tucked) clears the sentence, not Yes.",
    ],
  },
  {
    key: "thumbs_down",
    label: "Thumbs Down",
    how: "Thumbs down",
    meaning: "No",
    steps: [
      "Make a fist.",
      "Point the thumb straight down.",
      "Keep the other fingers curled.",
    ],
  },
  {
    key: "fist",
    label: "Fist",
    how: "Closed fist",
    meaning: "Clear",
    steps: [
      "Close every finger, including the thumb.",
      "Do not raise the thumb.",
      "Hold still. This erases the sentence so you can start over.",
    ],
  },
  {
    key: "four",
    label: "Four Fingers",
    how: "Four fingers, thumb in",
    meaning: "Undo",
    steps: [
      "Raise index, middle, ring, and pinky.",
      "Tuck the thumb against the palm (not out like Hello).",
      "Hold still. This removes only the last word.",
    ],
  },
  {
    key: "pointing",
    label: "Pointing",
    how: "Point",
    meaning: "Help",
    steps: [
      "Curl middle, ring, and pinky.",
      "Point the index finger at the camera.",
      "Keep the thumb tucked.",
    ],
  },
  {
    key: "peace_sign",
    label: "Peace Sign",
    how: "Peace",
    meaning: "Thank you",
    steps: [
      "Raise index and middle fingers in a V.",
      "Curl ring and pinky.",
      "Hold the V toward the camera.",
    ],
  },
  {
    key: "please",
    label: "Shaka",
    how: "Thumb and pinky",
    meaning: "Please",
    steps: [
      "Stick out the thumb and the pinky.",
      "Curl index, middle, and ring.",
      "Hold still.",
    ],
  },
  {
    key: "you",
    label: "Pinky",
    how: "Pinky up",
    meaning: "You",
    steps: [
      "Raise only the pinky.",
      "Keep thumb and the other fingers down.",
      "Hold still.",
    ],
  },
  {
    key: "want",
    label: "Three Fingers",
    how: "Three fingers",
    meaning: "Want",
    steps: [
      "Raise index, middle, and ring.",
      "Keep the pinky curled.",
      "Hold still.",
    ],
  },
  {
    key: "okay",
    label: "Okay Sign",
    how: "Okay (index curled)",
    meaning: "Okay",
    steps: [
      "Curl the index finger.",
      "Keep middle, ring, and pinky up.",
      "Hold still.",
    ],
  },
  {
    key: "i_love_you",
    label: "I Love You",
    how: "Thumb, index, pinky",
    meaning: "I love you",
    steps: [
      "Raise thumb, index, and pinky.",
      "Curl middle and ring.",
      "Hold still.",
    ],
  },
  {
    key: "wave",
    label: "Wave",
    how: "Wag left-right twice",
    meaning: "Goodbye",
    steps: [
      "Open your hand.",
      "Wag left, then right, then left (two direction changes).",
      "A still palm is Hello, not Goodbye.",
    ],
  },
] as const;

export type SignGesture = (typeof SIGN_VOCAB)[number]["key"];
export type SignMeaning = (typeof SIGN_VOCAB)[number]["meaning"];
export type SignVocabItem = (typeof SIGN_VOCAB)[number];

export const SIGN_GESTURES: SignGesture[] = SIGN_VOCAB.map((item) => item.key);

export const SIGN_MEANINGS: Record<SignGesture, SignMeaning> = Object.fromEntries(
  SIGN_VOCAB.map((item) => [item.key, item.meaning]),
) as Record<SignGesture, SignMeaning>;

export const SIGN_LABELS: Record<SignGesture, string> = Object.fromEntries(
  SIGN_VOCAB.map((item) => [item.key, item.label]),
) as Record<SignGesture, string>;

export function isSignGesture(value: string): value is SignGesture {
  return (SIGN_GESTURES as readonly string[]).includes(value);
}

export function isClearMeaning(value: string | null | undefined): boolean {
  return value === "Clear";
}

export function isUndoMeaning(value: string | null | undefined): boolean {
  return value === "Undo";
}
