import { SIGN_VOCAB, type SignMeaning } from "./signVocab";

export type { SignMeaning };

export const SIGN_MEANING_VALUES: readonly SignMeaning[] = SIGN_VOCAB.map(
  (item) => item.meaning,
);

export const MAX_SENTENCE_TOKENS = 10;
export const SENTENCE_SPEAK_IDLE_MS = 2200;

export function isSignMeaning(value: string): value is SignMeaning {
  return (SIGN_MEANING_VALUES as readonly string[]).includes(value);
}

function clauseFor(meaning: SignMeaning): string {
  switch (meaning) {
    case "Hello":
      return "Hello";
    case "Yes":
      return "Yes";
    case "No":
      return "No";
    case "Help":
      return "I need help";
    case "Thank you":
      return "Thank you";
    case "Goodbye":
      return "Goodbye";
    case "Please":
      return "Please";
    case "You":
      return "You";
    case "Want":
      return "I want that";
    case "Okay":
      return "Okay";
    case "I love you":
      return "I love you";
    case "Clear":
      return "";
    default: {
      const exhaustive: never = meaning;
      return exhaustive;
    }
  }
}

const PAIR_CLAUSES: Partial<Record<string, string>> = {
  "Hello|Help": "Hello, I need help",
  "Hello|You": "Hello, how are you",
  "Hello|Thank you": "Hello. Thank you",
  "Hello|Goodbye": "Hello. Goodbye",
  "Hello|I love you": "Hello. I love you",
  "Yes|Help": "Yes, I need help",
  "Yes|Please": "Yes, please",
  "Please|Help": "Please, I need help",
  "Please|You": "Please, I need you",
  "Please|Thank you": "Please, and thank you",
  "You|Help": "I need your help",
  "You|Thank you": "Thank you",
  "Want|Help": "I want help",
  "Want|You": "I want to talk to you",
  "Want|Please": "I want that, please",
  "No|Help": "I do not need help",
  "No|Thank you": "No, thank you",
  "Okay|Thank you": "Okay. Thank you",
  "Okay|Goodbye": "Okay. Goodbye",
  "Thank you|Goodbye": "Thank you. Goodbye",
  "I love you|You": "I love you",
};

const TRIPLE_CLAUSES: Partial<Record<string, string>> = {
  "Please|Want|Help": "Please, I want help",
  "Yes|Please|Help": "Yes, please help me",
  "Please|You|Help": "Please, I need your help",
  "Hello|I love you|You": "Hello. I love you",
};

/**
 * Turns a sequence of recognized sign meanings into one spoken sentence.
 * Common everyday combinations are merged so it sounds like a person talking.
 */
export function composeSignSentence(meanings: readonly string[]): string {
  const tokens = meanings.filter(isSignMeaning);
  if (tokens.length === 0) return "";

  const clauses: string[] = [];
  let index = 0;

  while (index < tokens.length) {
    const first = tokens[index];
    if (!first) {
      break;
    }
    const second = tokens[index + 1];
    const third = tokens[index + 2];

    if (second && third) {
      const triple = TRIPLE_CLAUSES[`${first}|${second}|${third}`];
      if (triple) {
        clauses.push(triple);
        index += 3;
        continue;
      }
    }

    if (second) {
      const pair = PAIR_CLAUSES[`${first}|${second}`];
      if (pair) {
        clauses.push(pair);
        index += 2;
        continue;
      }
    }

    clauses.push(clauseFor(first));
    index += 1;
  }

  const spoken = clauses.filter((clause) => clause.length > 0);
  if (spoken.length === 0) return "";
  const text = spoken.join(". ");
  return text.endsWith(".") ? text : `${text}.`;
}

export function appendSignToken(
  tokens: readonly string[],
  meaning: string,
): string[] {
  if (meaning === "Clear") return [];
  if (!isSignMeaning(meaning)) return [...tokens];
  if (tokens.length >= MAX_SENTENCE_TOKENS) return [...tokens];
  return [...tokens, meaning];
}
