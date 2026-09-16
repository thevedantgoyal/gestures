"use client";

import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { SignCommunicator } from "../components/SignCommunicator";
import { SignRulebook } from "../components/SignRulebook";
import { SignTrainer } from "../components/SignTrainer";
import {
  appendSignToken,
  composeSignSentence,
  SENTENCE_SPEAK_IDLE_MS,
} from "../lib/signSentence";
import {
  isSignTrainStats,
  liveTestFromDebug,
  type SignTrainStats,
} from "../lib/signTrain";
import {
  drawHandOverlay,
  isTrailMoving,
  pushFingertipTrail,
  pushIndexTrail,
  TRAIL_MS,
  type HandDrawStyle,
  type TrailPoint,
} from "../lib/handOverlay";
import {
  isClearMeaning,
  isSignGesture,
  SIGN_GESTURES,
  SIGN_LABELS,
  SIGN_MEANINGS,
  type SignGesture,
} from "../lib/signVocab";

type Mode = "aviation" | "sign_language";
type WsStatus =
  | "idle"
  | "connecting"
  | "connected"
  | "disconnected"
  | "reconnecting";
type AviationGesture = "exit_pointing" | "seatbelt_demo";
type CalibrationGesture = AviationGesture | SignGesture;
type CalibrationPhase =
  | "idle"
  | "countdown"
  | "recording"
  | "saving"
  | "saved"
  | "error";
type ReferenceStatus = Record<string, "recorded" | "default">;

const ACCENT = "#2C6E8C";
const WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const POSE_MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const HAND_MODEL_PATH =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const API_BASE = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/landmarks";
const WS_SEND_INTERVAL_MS = 120;
const WRIST_VISIBILITY_THRESHOLD = 0.3;
const SPEAK_DWELL_MS = 400;
const AVIATION_FEEDBACK_DEBOUNCE_MS = 500;
const RECORD_CAPTURE_INTERVAL_MS = 100;
const WS_MAX_RECONNECT_ATTEMPTS = 5;
const WS_RECONNECT_DELAY_MS = 1500;
const COACHING_UI_TIMEOUT_MS = 5000;
const COACHING_FALLBACK_TEXT = "Adjust your arm position and try again.";

const RECORD_DURATION_MS: Record<CalibrationGesture, number> = {
  exit_pointing: 3000,
  seatbelt_demo: 2000,
  thumbs_up: 2500,
  thumbs_down: 2500,
  open_palm: 2500,
  fist: 2500,
  pointing: 2500,
  peace_sign: 2500,
  please: 2500,
  you: 2500,
  want: 2500,
  okay: 2500,
  i_love_you: 2500,
  wave: 2500,
};

const AVIATION_GESTURES: AviationGesture[] = ["exit_pointing", "seatbelt_demo"];

const GESTURE_LABELS: Record<CalibrationGesture, string> = {
  exit_pointing: "Exit Pointing",
  seatbelt_demo: "Seatbelt Demo",
  ...SIGN_LABELS,
};

const LANDMARK_INDEX = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
} as const;

const HAND_LANDMARK_KEYS = [
  "wrist",
  "thumb_cmc",
  "thumb_mcp",
  "thumb_ip",
  "thumb_tip",
  "index_mcp",
  "index_pip",
  "index_dip",
  "index_tip",
  "middle_mcp",
  "middle_pip",
  "middle_dip",
  "middle_tip",
  "ring_mcp",
  "ring_pip",
  "ring_dip",
  "ring_tip",
  "pinky_mcp",
  "pinky_pip",
  "pinky_dip",
  "pinky_tip",
] as const;

type LandmarkPoint = {
  x: number;
  y: number;
  z: number;
  visibility: number;
};

type PoseStreamLandmarks = {
  leftShoulder: LandmarkPoint | null;
  rightShoulder: LandmarkPoint | null;
  leftElbow: LandmarkPoint | null;
  rightElbow: LandmarkPoint | null;
  leftWrist: LandmarkPoint | null;
  rightWrist: LandmarkPoint | null;
  leftHip: LandmarkPoint | null;
  rightHip: LandmarkPoint | null;
};

type HandStreamLandmarks = {
  [K in (typeof HAND_LANDMARK_KEYS)[number]]: LandmarkPoint | null;
} & {
  handedness?: string;
};

type StreamLandmarks = PoseStreamLandmarks | HandStreamLandmarks;

type GestureFeedback = {
  gesture: CalibrationGesture | "none";
  correct?: boolean;
  score: number;
  meaning?: string | null;
  left_arm_angle?: number | null;
  right_arm_angle?: number | null;
  target_angle?: number | null;
  tolerance?: number | null;
  pass_band_deg?: number | null;
  hints?: string[];
  reason?: string | null;
};

function sleep(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

function toLandmarkPoint(
  landmark: NormalizedLandmark | undefined,
): LandmarkPoint | null {
  if (!landmark) return null;
  return {
    x: landmark.x,
    y: landmark.y,
    z: landmark.z,
    // PoseLandmarker sets visibility; HandLandmarker often omits it or sends 0.
    visibility: landmark.visibility ?? 1,
  };
}

function assessHandFramingClient(landmarks: HandStreamLandmarks): {
  ok: boolean;
  reason: string;
  pointsInFrame: number;
  bboxArea: number;
  sampleVisibility: number | null | undefined;
} {
  const margin = 0.02;
  const minArea = 0.008;
  const xs: number[] = [];
  const ys: number[] = [];
  let inFrame = 0;
  let sampleVisibility: number | null | undefined;

  for (const key of HAND_LANDMARK_KEYS) {
    const point = landmarks[key];
    if (!point) continue;
    xs.push(point.x);
    ys.push(point.y);
    if (sampleVisibility === undefined) {
      sampleVisibility = point.visibility;
    }
    if (
      point.x >= margin &&
      point.x <= 1 - margin &&
      point.y >= margin &&
      point.y <= 1 - margin
    ) {
      inFrame += 1;
    }
  }

  if (xs.length < HAND_LANDMARK_KEYS.length) {
    return {
      ok: false,
      reason: "incomplete_landmarks",
      pointsInFrame: inFrame,
      bboxArea: 0,
      sampleVisibility,
    };
  }

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const bboxArea = Math.max(0, maxX - minX) * Math.max(0, maxY - minY);
  const allIn = inFrame >= HAND_LANDMARK_KEYS.length;
  const sizeOk = bboxArea >= minArea;
  const ok = allIn && sizeOk;

  return {
    ok,
    reason: !allIn ? "landmarks_near_edge" : !sizeOk ? "bbox_too_small" : "ok",
    pointsInFrame: inFrame,
    bboxArea,
    sampleVisibility,
  };
}

function extractPoseLandmarks(
  landmarks: NormalizedLandmark[],
): PoseStreamLandmarks {
  return {
    leftShoulder: toLandmarkPoint(landmarks[LANDMARK_INDEX.leftShoulder]),
    rightShoulder: toLandmarkPoint(landmarks[LANDMARK_INDEX.rightShoulder]),
    leftElbow: toLandmarkPoint(landmarks[LANDMARK_INDEX.leftElbow]),
    rightElbow: toLandmarkPoint(landmarks[LANDMARK_INDEX.rightElbow]),
    leftWrist: toLandmarkPoint(landmarks[LANDMARK_INDEX.leftWrist]),
    rightWrist: toLandmarkPoint(landmarks[LANDMARK_INDEX.rightWrist]),
    leftHip: toLandmarkPoint(landmarks[LANDMARK_INDEX.leftHip]),
    rightHip: toLandmarkPoint(landmarks[LANDMARK_INDEX.rightHip]),
  };
}

function extractHandLandmarks(
  landmarks: NormalizedLandmark[],
  handedness?: string,
): HandStreamLandmarks {
  const payload = {} as HandStreamLandmarks;
  HAND_LANDMARK_KEYS.forEach((key, index) => {
    payload[key] = toLandmarkPoint(landmarks[index]);
  });
  if (handedness) payload.handedness = handedness;
  return payload;
}

/** Prefer the largest detected person/hand to reduce flicker with multiple subjects. */
function pickLargestLandmarks(
  groups: NormalizedLandmark[][],
): { landmarks: NormalizedLandmark[]; index: number } | null {
  if (!groups.length) return null;
  let bestIndex = 0;
  let bestArea = -1;
  groups.forEach((group, index) => {
    let minX = 1;
    let minY = 1;
    let maxX = 0;
    let maxY = 0;
    for (const point of group) {
      minX = Math.min(minX, point.x);
      minY = Math.min(minY, point.y);
      maxX = Math.max(maxX, point.x);
      maxY = Math.max(maxY, point.y);
    }
    const area = Math.max(0, maxX - minX) * Math.max(0, maxY - minY);
    if (area > bestArea) {
      bestArea = area;
      bestIndex = index;
    }
  });
  return { landmarks: groups[bestIndex], index: bestIndex };
}

function cameraPermissionMessage(err: unknown): string {
  const name =
    err && typeof err === "object" && "name" in err
      ? String((err as { name?: string }).name)
      : "";
  if (
    name === "NotAllowedError" ||
    name === "PermissionDeniedError" ||
    (err instanceof Error && /permission|denied|not allowed/i.test(err.message))
  ) {
    return "Camera access is needed to use this — please allow it and refresh";
  }
  if (name === "NotFoundError") {
    return "No camera was found. Connect a webcam and try again.";
  }
  if (err instanceof Error && err.message) return err.message;
  return "Unable to access webcam";
}

function drawPose(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  width: number,
  height: number,
) {
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = ACCENT;
  ctx.fillStyle = ACCENT;
  ctx.lineWidth = 1.5;
  ctx.lineCap = "round";

  for (const connection of PoseLandmarker.POSE_CONNECTIONS) {
    const start = landmarks[connection.start];
    const end = landmarks[connection.end];
    if (!start || !end) continue;

    ctx.beginPath();
    ctx.moveTo(start.x * width, start.y * height);
    ctx.lineTo(end.x * width, end.y * height);
    ctx.stroke();
  }

  for (const landmark of landmarks) {
    ctx.beginPath();
    ctx.arc(landmark.x * width, landmark.y * height, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

const SPEECH_CLIPS: Record<string, string> = {
  Ready: "/speech/ready.wav",
  Yes: "/speech/yes.wav",
  Hello: "/speech/hello.wav",
  Stop: "/speech/stop.wav",
  Help: "/speech/help.wav",
  "Thank you": "/speech/thank-you.wav",
  Goodbye: "/speech/goodbye.wav",
};

let speechCtx: AudioContext | null = null;
const speechBuffers = new Map<string, AudioBuffer>();
let speechBusy = false;
let activeSource: AudioBufferSourceNode | null = null;

function stopSpeechPlayback() {
  try {
    activeSource?.stop();
  } catch {
    // already stopped
  }
  activeSource = null;
  speechBusy = false;
}

function stopAllSpeech() {
  stopSpeechPlayback();
  if (typeof window !== "undefined" && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
}

function pickEnglishVoice(): SpeechSynthesisVoice | null {
  if (typeof window === "undefined" || !window.speechSynthesis) return null;
  const voices = window.speechSynthesis.getVoices();
  if (voices.length === 0) return null;
  return (
    voices.find(
      (voice) =>
        /en-US/i.test(voice.lang) &&
        /google|microsoft|samantha|aria|natural/i.test(voice.name),
    ) ??
    voices.find((voice) => /en-US/i.test(voice.lang)) ??
    voices.find((voice) => /^en(-|$)/i.test(voice.lang)) ??
    null
  );
}

function speakSentence(
  text: string,
  hooks?: { onStart?: () => void; onEnd?: () => void },
): boolean {
  const trimmed = text.trim();
  if (!trimmed) return false;
  if (typeof window === "undefined" || !("speechSynthesis" in window)) {
    return speakMeaning(trimmed, hooks);
  }

  stopSpeechPlayback();
  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(trimmed);
  utterance.rate = 0.92;
  utterance.pitch = 1;
  utterance.lang = "en-US";
  const voice = pickEnglishVoice();
  if (voice) utterance.voice = voice;
  utterance.onstart = () => hooks?.onStart?.();
  utterance.onend = () => hooks?.onEnd?.();
  utterance.onerror = () => hooks?.onEnd?.();
  window.speechSynthesis.speak(utterance);
  return true;
}

async function loadSpeechBuffers(ctx: AudioContext) {
  const missing = Object.entries(SPEECH_CLIPS).filter(
    ([phrase]) => !speechBuffers.has(phrase),
  );
  await Promise.all(
    missing.map(async ([phrase, url]) => {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to load ${url} (${response.status})`);
      }
      const bytes = await response.arrayBuffer();
      const buffer = await ctx.decodeAudioData(bytes.slice(0));
      speechBuffers.set(phrase, buffer);
    }),
  );
}

function playPhrase(
  phrase: string,
  hooks?: { onStart?: () => void; onEnd?: () => void },
): boolean {
  if (!speechCtx || speechCtx.state === "closed") {
    console.warn("[speech] audio not unlocked yet");
    return false;
  }
  if (speechBusy) {
    console.log("[speech] skip (already playing):", phrase);
    return false;
  }
  const buffer = speechBuffers.get(phrase);
  if (!buffer) {
    console.warn("[speech] missing clip:", phrase);
    return false;
  }
  if (speechCtx.state === "suspended") {
    void speechCtx.resume();
  }

  const source = speechCtx.createBufferSource();
  const gain = speechCtx.createGain();
  gain.gain.value = 1;
  source.buffer = buffer;
  source.connect(gain);
  gain.connect(speechCtx.destination);

  speechBusy = true;
  activeSource = source;
  source.onended = () => {
    if (activeSource === source) activeSource = null;
    speechBusy = false;
    console.log("[speech] ended:", phrase);
    hooks?.onEnd?.();
  };

  console.log("[speech] playing clip:", phrase);
  hooks?.onStart?.();
  source.start(0);
  return true;
}

function speakMeaning(
  text: string,
  hooks?: { onStart?: () => void; onEnd?: () => void },
): boolean {
  return playPhrase(text.trim(), hooks);
}

function unlockSpeechSynthesis(
  hooks?: { onStart?: () => void; onEnd?: () => void },
): boolean {
  if (typeof window === "undefined") return false;
  const AudioCtx =
    window.AudioContext ||
    (
      window as unknown as {
        webkitAudioContext?: typeof AudioContext;
      }
    ).webkitAudioContext;
  if (!AudioCtx) {
    console.warn("[speech] Web Audio API unavailable");
    return false;
  }

  if (!speechCtx || speechCtx.state === "closed") {
    speechCtx = new AudioCtx();
  }

  console.log("[speech] unlock: resuming audio context + playing Ready");
  void speechCtx
    .resume()
    .then(async () => {
      if (!speechCtx) return;
      await loadSpeechBuffers(speechCtx);
      playPhrase("Ready", hooks);
    })
    .catch((err) => {
      console.warn("[speech] unlock failed:", err);
    });
  return true;
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("aviation");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [poseReady, setPoseReady] = useState(false);
  const [handReady, setHandReady] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  const [debugOpen, setDebugOpen] = useState(false);
  const [debugLandmarks, setDebugLandmarks] = useState<StreamLandmarks | null>(
    null,
  );
  const [wsStatus, setWsStatus] = useState<WsStatus>("idle");
  const [lastRoundTripMs, setLastRoundTripMs] = useState<number | null>(null);
  const [gestureFeedback, setGestureFeedback] =
    useState<GestureFeedback | null>(null);
  const [limbsNotVisible, setLimbsNotVisible] = useState(false);
  const [scoreDebug, setScoreDebug] = useState<Record<string, unknown> | null>(
    null,
  );
  const [coachingText, setCoachingText] = useState<string | null>(null);
  const [coachingLoading, setCoachingLoading] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [voiceUnlocked, setVoiceUnlocked] = useState(false);
  const [motionProgress, setMotionProgress] = useState<number | null>(null);
  const [motionHint, setMotionHint] = useState<string | null>(null);
  const [handStyle, setHandStyle] = useState<HandDrawStyle | null>(null);
  const [calibrationOpen, setCalibrationOpen] = useState(false);
  const [trainOpen, setTrainOpen] = useState(false);
  const [mlStats, setMlStats] = useState<SignTrainStats | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainBusy, setTrainBusy] = useState(false);
  const [selectedTrainGesture, setSelectedTrainGesture] =
    useState<SignGesture>("open_palm");
  const [rulebookOpen, setRulebookOpen] = useState(false);
  const [selectedRulebookGesture, setSelectedRulebookGesture] =
    useState<SignGesture>("open_palm");
  const [calibrationPhase, setCalibrationPhase] =
    useState<CalibrationPhase>("idle");
  const [calibrationGesture, setCalibrationGesture] =
    useState<CalibrationGesture | null>(null);
  const [calibrationMessage, setCalibrationMessage] = useState<string | null>(
    null,
  );
  const [countdown, setCountdown] = useState(3);
  const [recordProgress, setRecordProgress] = useState(0);
  const [referenceStatus, setReferenceStatus] =
    useState<ReferenceStatus | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastDetectTimeRef = useRef(-1);
  const lastDebugUpdateRef = useRef(0);
  const lastWsSendRef = useRef(0);
  const wsRef = useRef<WebSocket | null>(null);
  const pendingSendAtRef = useRef<Map<number, number>>(new Map());
  const recorderRef = useRef<{
    active: boolean;
    frames: { timestamp: number; landmarks: StreamLandmarks }[];
    lastCaptureAt: number;
  }>({ active: false, frames: [], lastCaptureAt: 0 });
  const calibrationTokenRef = useRef(0);
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const pendingSpeakRef = useRef<{ gesture: string; since: number } | null>(
    null,
  );
  const sentenceTokensRef = useRef<string[]>([]);
  const lastCommittedSignRef = useRef<string | null>(null);
  const releasedAfterCommitRef = useRef(true);
  const [sentenceTokens, setSentenceTokens] = useState<string[]>([]);
  const [lastSpokenSentence, setLastSpokenSentence] = useState<string | null>(
    null,
  );
  const sessionIdRef = useRef<string>("");
  const motionHintTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const voiceUnlockedRef = useRef(false);
  voiceUnlockedRef.current = voiceUnlocked;
  const handLandmarkStructureLoggedRef = useRef(false);
  const indexTrailRef = useRef<TrailPoint[]>([]);
  const fingertipTrailRef = useRef<TrailPoint[]>([]);
  const handStyleRef = useRef<HandDrawStyle | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaChunksRef = useRef<Blob[]>([]);
  const sessionTimelineRef = useRef<
    {
      t_ms: number;
      gesture: string;
      correct: boolean | null;
      score: number;
      coaching_text?: string | null;
    }[]
  >([]);
  const recordingStartedAtRef = useRef<number | null>(null);
  const lastTimelineKeyRef = useRef<string>("");
  const aviationDebounceRef = useRef<{
    key: string;
    since: number;
  } | null>(null);

  const modelReady = mode === "aviation" ? poseReady : handReady;
  const composedSentence = composeSignSentence(sentenceTokens);

  function resetSentenceBuilder() {
    sentenceTokensRef.current = [];
    lastCommittedSignRef.current = null;
    releasedAfterCommitRef.current = true;
    setSentenceTokens([]);
    setLastSpokenSentence(null);
    setSpeaking(false);
    stopAllSpeech();
  }

  function speakComposedSentence(text: string) {
    const spoken = text.trim();
    if (!spoken) return;
    setLastSpokenSentence(spoken);
    speakSentence(spoken, {
      onStart: () => setSpeaking(true),
      onEnd: () => setSpeaking(false),
    });
    sentenceTokensRef.current = [];
    lastCommittedSignRef.current = null;
    releasedAfterCommitRef.current = true;
    setSentenceTokens([]);
  }

  useEffect(() => {
    if (mode !== "sign_language" || !isStreaming || !voiceUnlocked) return;
    if (sentenceTokens.length === 0) return;
    if (speaking) return;
    // Keep collecting signs while the current one is still locked.
    if (gestureFeedback?.correct) return;
    const text = composeSignSentence(sentenceTokens);
    const timer = window.setTimeout(() => {
      speakComposedSentence(text);
    }, SENTENCE_SPEAK_IDLE_MS);
    return () => window.clearTimeout(timer);
  }, [
    sentenceTokens,
    mode,
    isStreaming,
    voiceUnlocked,
    speaking,
    gestureFeedback?.correct,
  ]);

  useEffect(() => {
    let cancelled = false;

    async function loadModels() {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

        async function createPose(delegate: "GPU" | "CPU") {
          return PoseLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: POSE_MODEL_PATH, delegate },
            runningMode: "VIDEO",
            // Detect up to 2, then we pick the largest by bounding box.
            numPoses: 2,
          });
        }

        async function createHand(delegate: "GPU" | "CPU") {
          return HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: HAND_MODEL_PATH, delegate },
            runningMode: "VIDEO",
            numHands: 2,
          });
        }

        let pose: PoseLandmarker;
        try {
          pose = await createPose("GPU");
        } catch {
          pose = await createPose("CPU");
        }

        let hand: HandLandmarker;
        try {
          hand = await createHand("GPU");
        } catch {
          hand = await createHand("CPU");
        }

        if (cancelled) {
          pose.close();
          hand.close();
          return;
        }

        poseLandmarkerRef.current = pose;
        handLandmarkerRef.current = hand;
        setPoseReady(true);
        setHandReady(true);
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : "Failed to load detection models";
        if (!cancelled) setModelError(message);
      }
    }

    loadModels();

    return () => {
      cancelled = true;
      poseLandmarkerRef.current?.close();
      poseLandmarkerRef.current = null;
      handLandmarkerRef.current?.close();
      handLandmarkerRef.current = null;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      wsRef.current?.close();
      wsRef.current = null;
      if (typeof window !== "undefined") {
        stopAllSpeech();
      }
    };
  }, []);

  useEffect(() => {
    if (!isStreaming) {
      wsRef.current?.close();
      wsRef.current = null;
      pendingSendAtRef.current.clear();
      setWsStatus("idle");
      setLastRoundTripMs(null);
      setGestureFeedback(null);
      setLimbsNotVisible(false);
      setScoreDebug(null);
      setCoachingText(null);
      setCoachingLoading(false);
      setSpeaking(false);
      pendingSpeakRef.current = null;
      resetSentenceBuilder();
      return;
    }

    let disposed = false;
    let reconnectAttempts = 0;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const handleMessage = (event: MessageEvent) => {
      if (disposed) return;

      try {
        const data = JSON.parse(event.data) as {
          status?: string;
          timestamp?: number;
          gesture?: string;
          correct?: boolean;
          score?: number;
          meaning?: string;
          reason?: string;
          attempted_gesture?: string;
          hint?: string;
          buffer_progress?: number;
          scores?: Record<string, number>;
          framing?: Record<string, unknown>;
          classified_as?: string;
          left_arm_angle?: number | null;
          right_arm_angle?: number | null;
          target_angle?: number;
          tolerance?: number;
          pass_band_deg?: number;
          hints?: string[];
          deviations?: unknown;
          left_wrist_visibility?: number;
          right_wrist_visibility?: number;
          dtw_distance?: number;
          coaching_text?: string | null;
          coaching_pending?: boolean;
          motion_active?: boolean;
          motion_energy?: number;
          hand_style?: HandDrawStyle;
          recognizer?: string;
          model?: {
            available?: boolean;
            version?: string | null;
            label?: string | null;
            prob?: number | null;
            agree?: boolean | null;
          };
        };

        if (data.status === "received" && typeof data.timestamp === "number") {
          const sentAt = pendingSendAtRef.current.get(data.timestamp);
          if (sentAt !== undefined) {
            setLastRoundTripMs(Math.round(performance.now() - sentAt));
            pendingSendAtRef.current.delete(data.timestamp);
          }
          setWsStatus("connected");
        }

        if (
          (data.reason === "motion_buffering" ||
            data.reason === "motion_in_progress") &&
          typeof data.buffer_progress === "number"
        ) {
          setMotionProgress(
            Math.max(0, Math.min(1, data.buffer_progress)),
          );
        } else if (
          data.reason === "motion_in_progress" ||
          data.reason === "motion_buffering"
        ) {
          setMotionProgress((prev) => prev ?? 0.45);
        } else if (data.gesture && data.gesture !== "none") {
          setMotionProgress(null);
        } else if (
          data.reason !== "motion_buffering" &&
          data.reason !== "motion_in_progress"
        ) {
          setMotionProgress(null);
        }

        if (
          typeof data.hint === "string" &&
          data.hint &&
          (data.reason === "not_recognized" ||
            data.reason === "motion_in_progress" ||
            data.reason === "motion_buffering")
        ) {
          setMotionHint(data.hint);
          if (motionHintTimerRef.current) {
            clearTimeout(motionHintTimerRef.current);
          }
          motionHintTimerRef.current = setTimeout(() => {
            setMotionHint(null);
            motionHintTimerRef.current = null;
          }, 4000);
        }

        setScoreDebug({
          gesture: data.gesture ?? "none",
          correct: data.correct ?? null,
          score: data.score ?? 0,
          meaning: data.meaning ?? null,
          reason: data.reason ?? null,
          attempted_gesture: data.attempted_gesture ?? null,
          hint: data.hint ?? null,
          buffer_progress: data.buffer_progress ?? null,
          scores: data.scores ?? null,
          framing: data.framing ?? null,
          classified_as: data.classified_as ?? null,
          left_arm_angle: data.left_arm_angle ?? null,
          right_arm_angle: data.right_arm_angle ?? null,
          target_angle: data.target_angle ?? null,
          tolerance: data.tolerance ?? null,
          left_wrist_visibility: data.left_wrist_visibility ?? null,
          right_wrist_visibility: data.right_wrist_visibility ?? null,
          dtw_distance: data.dtw_distance ?? null,
          deviations: data.deviations ?? [],
          coaching_text: data.coaching_text ?? null,
          coaching_pending: data.coaching_pending ?? false,
          motion_active: data.motion_active ?? null,
          motion_energy: data.motion_energy ?? null,
          hand_style: data.hand_style ?? null,
          recognizer: data.recognizer ?? null,
          model: data.model ?? null,
        });

        const currentMode = modeRef.current;

        if (currentMode === "sign_language") {
          const gesture = data.gesture ?? "none";
          const isSign = isSignGesture(gesture);
          const isCorrect =
            Boolean(data.correct) && typeof data.meaning === "string";
          const score = typeof data.score === "number" ? data.score : 0;
          const keepMotionBar =
            data.reason === "motion_buffering" ||
            data.reason === "motion_in_progress";

          if (gesture === "none") {
            releasedAfterCommitRef.current = true;
          }

          if (isSign) {
            setGestureFeedback({
              gesture,
              correct: isCorrect,
              score,
              meaning: isCorrect ? data.meaning : null,
            });
            setCoachingText(null);
            setCoachingLoading(false);
            if (!keepMotionBar) {
              setMotionProgress(null);
            }

            if (isCorrect && typeof data.meaning === "string") {
              const skipPalmAfterWave =
                gesture === "open_palm" &&
                lastCommittedSignRef.current === "wave" &&
                !releasedAfterCommitRef.current;
              if (skipPalmAfterWave) {
                pendingSpeakRef.current = null;
              } else {
                const now = Date.now();
                if (pendingSpeakRef.current?.gesture !== gesture) {
                  pendingSpeakRef.current = { gesture, since: now };
                }
                const heldMs = now - pendingSpeakRef.current.since;
                if (
                  heldMs >= SPEAK_DWELL_MS &&
                  lastCommittedSignRef.current !== gesture
                ) {
                  lastCommittedSignRef.current = gesture;
                  releasedAfterCommitRef.current = false;
                  if (gesture === "fist" || isClearMeaning(data.meaning)) {
                    resetSentenceBuilder();
                    lastCommittedSignRef.current = "fist";
                    setGestureFeedback({
                      gesture: "fist",
                      correct: true,
                      score,
                      meaning: "Clear",
                    });
                  } else {
                    const next = appendSignToken(
                      sentenceTokensRef.current,
                      data.meaning,
                    );
                    sentenceTokensRef.current = next;
                    setSentenceTokens(next);
                  }
                }
              }
            } else if (pendingSpeakRef.current?.gesture !== gesture) {
              pendingSpeakRef.current = null;
            }
          } else if (data.status === "received") {
            pendingSpeakRef.current = null;
            setGestureFeedback({
              gesture: "none",
              correct: false,
              score,
              meaning: null,
            });
            if (!keepMotionBar) {
              setMotionProgress(null);
            }
          }
          return;
        }

        if (
          data.gesture === "exit_pointing" ||
          data.gesture === "seatbelt_demo"
        ) {
          const isCorrect = Boolean(data.correct);
          const hints = Array.isArray(data.hints)
            ? data.hints.filter((h): h is string => typeof h === "string")
            : [];
          const debounceKey = `${data.gesture}:${isCorrect}`;
          const now = Date.now();
          if (
            !aviationDebounceRef.current ||
            aviationDebounceRef.current.key !== debounceKey
          ) {
            aviationDebounceRef.current = { key: debounceKey, since: now };
          }
          // Require stable gesture+correct for ~500ms before updating the badge.
          if (
            now - aviationDebounceRef.current.since <
            AVIATION_FEEDBACK_DEBOUNCE_MS
          ) {
            return;
          }
          setGestureFeedback({
            gesture: data.gesture,
            correct: isCorrect,
            score: typeof data.score === "number" ? data.score : 0,
            left_arm_angle: data.left_arm_angle ?? null,
            right_arm_angle: data.right_arm_angle ?? null,
            target_angle: data.target_angle ?? null,
            tolerance: data.tolerance ?? null,
            pass_band_deg: data.pass_band_deg ?? null,
            hints,
            reason: data.reason ?? null,
          });
          pushTimelineEvent({
            gesture: data.gesture,
            correct: isCorrect,
            score: typeof data.score === "number" ? data.score : 0,
            coaching_text:
              typeof data.coaching_text === "string" ? data.coaching_text : null,
          });
          setMotionProgress(null);

          if (isCorrect) {
            setCoachingText(null);
            setCoachingLoading(false);
          } else if (
            typeof data.coaching_text === "string" &&
            data.coaching_text
          ) {
            setCoachingText(data.coaching_text);
            setCoachingLoading(false);
          } else if (data.coaching_pending) {
            setCoachingLoading(true);
          } else if (hints.length > 0) {
            setCoachingText(`You're doing it wrong. ${hints[0]}.`);
            setCoachingLoading(false);
          }
        } else if (
          data.status === "received" &&
          (data.reason === "idle" ||
            data.reason === "no_attempt" ||
            data.reason === "low_visibility" ||
            data.gesture === "none")
        ) {
          // Standing still / no attempt — clear to neutral, don't flash Incorrect.
          aviationDebounceRef.current = null;
          setGestureFeedback({
            gesture: "none",
            correct: false,
            score: 0,
            reason: data.reason ?? "idle",
          });
          if (data.reason !== "motion_buffering") {
            setCoachingText(null);
            setCoachingLoading(false);
          }
        }
      } catch {
        // Ignore malformed acknowledgments — keep the camera loop running.
      }
    };

    function connect() {
      if (disposed) return;
      setWsStatus(reconnectAttempts === 0 ? "connecting" : "reconnecting");

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (disposed) {
          ws.close();
          return;
        }
        reconnectAttempts = 0;
        setWsStatus("connected");
      };

      ws.onmessage = handleMessage;

      ws.onerror = () => {
        // Browser follows with onclose; keep UI on Disconnected / reconnecting.
      };

      ws.onclose = () => {
        if (disposed) return;
        if (wsRef.current === ws) wsRef.current = null;

        if (reconnectAttempts < WS_MAX_RECONNECT_ATTEMPTS) {
          reconnectAttempts += 1;
          setWsStatus("reconnecting");
          reconnectTimer = setTimeout(connect, WS_RECONNECT_DELAY_MS);
        } else {
          setWsStatus("disconnected");
        }
      };
    }

    connect();

    return () => {
      disposed = true;
      if (reconnectTimer !== null) clearTimeout(reconnectTimer);
      wsRef.current?.close();
      if (wsRef.current) wsRef.current = null;
    };
  }, [isStreaming]);

  // If coaching never arrives (API hang / dropped follow-up), fall back after 5s.
  useEffect(() => {
    if (!coachingLoading) return;
    const timer = window.setTimeout(() => {
      setCoachingLoading(false);
      setCoachingText((prev) => prev ?? COACHING_FALLBACK_TEXT);
    }, COACHING_UI_TIMEOUT_MS);
    return () => window.clearTimeout(timer);
  }, [coachingLoading]);

  // Clear live results when switching modes so pose/hand state can't stick.
  useEffect(() => {
    setGestureFeedback(null);
    setCoachingText(null);
    setCoachingLoading(false);
    setScoreDebug(null);
    setLimbsNotVisible(false);
    setSpeaking(false);
    setMotionProgress(null);
    setMotionHint(null);
    setHandStyle(null);
    indexTrailRef.current = [];
    fingertipTrailRef.current = [];
    handStyleRef.current = null;
    if (motionHintTimerRef.current) {
      clearTimeout(motionHintTimerRef.current);
      motionHintTimerRef.current = null;
    }
    pendingSpeakRef.current = null;
    resetSentenceBuilder();
    lastDetectTimeRef.current = -1;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, [mode]);

  useEffect(() => {
    const canDetect = isStreaming && modelReady;

    if (!canDetect) {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!isStreaming) {
        setDebugLandmarks(null);
        setLimbsNotVisible(false);
        setScoreDebug(null);
        setCoachingText(null);
        setCoachingLoading(false);
        lastDetectTimeRef.current = -1;
        lastWsSendRef.current = 0;
      }
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const predict = () => {
      const ctx = canvas.getContext("2d");
      const activeMode = modeRef.current;

      if (
        ctx &&
        video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        video.currentTime !== lastDetectTimeRef.current
      ) {
        lastDetectTimeRef.current = video.currentTime;

        if (
          canvas.width !== video.videoWidth ||
          canvas.height !== video.videoHeight
        ) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        const now = performance.now();
        let streamLandmarks: StreamLandmarks | null = null;
        let notVisible = false;

        if (activeMode === "sign_language") {
          const handLandmarker = handLandmarkerRef.current;
          if (handLandmarker) {
            try {
              const result = handLandmarker.detectForVideo(
                video,
                performance.now(),
              );
              const picked = pickLargestLandmarks(result.landmarks);
              if (picked) {
                if (!handLandmarkStructureLoggedRef.current) {
                  handLandmarkStructureLoggedRef.current = true;
                  const sample = picked.landmarks[0] as NormalizedLandmark & {
                    visibility?: number;
                  };
                  console.log(
                    "[hand] raw landmark[0] keys:",
                    sample ? Object.keys(sample) : null,
                    "visibility=",
                    sample?.visibility,
                    "full sample=",
                    sample,
                  );
                }
                const nowMs = performance.now();
                indexTrailRef.current = pushIndexTrail(
                  indexTrailRef.current,
                  picked.landmarks,
                  nowMs,
                );
                const moving = isTrailMoving(indexTrailRef.current);
                if (moving) {
                  fingertipTrailRef.current = pushFingertipTrail(
                    fingertipTrailRef.current,
                    picked.landmarks,
                    nowMs,
                  );
                } else {
                  fingertipTrailRef.current = fingertipTrailRef.current.filter(
                    (point) => nowMs - point.t <= TRAIL_MS,
                  );
                }
                const style: HandDrawStyle = moving ? "motion" : "hold";
                if (handStyleRef.current !== style) {
                  handStyleRef.current = style;
                  setHandStyle(style);
                }
                drawHandOverlay(
                  ctx,
                  [picked.landmarks],
                  fingertipTrailRef.current,
                  canvas.width,
                  canvas.height,
                  style,
                );
                const handedness =
                  result.handednesses?.[picked.index]?.[0]?.categoryName ??
                  undefined;
                streamLandmarks = extractHandLandmarks(
                  picked.landmarks,
                  handedness,
                );
                const framing = assessHandFramingClient(
                  streamLandmarks as HandStreamLandmarks,
                );
                notVisible = !framing.ok;
                if (now - lastDebugUpdateRef.current > 500) {
                  console.log(
                    "[hand_framing]",
                    framing.reason,
                    "in_frame=",
                    framing.pointsInFrame,
                    "bbox_area=",
                    framing.bboxArea.toFixed(4),
                    "sample_visibility=",
                    framing.sampleVisibility,
                  );
                }
              } else {
                indexTrailRef.current = [];
                fingertipTrailRef.current = [];
                if (handStyleRef.current !== null) {
                  handStyleRef.current = null;
                  setHandStyle(null);
                }
                ctx.clearRect(0, 0, canvas.width, canvas.height);
              }
            } catch {
              ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
          }
        } else {
          const poseLandmarker = poseLandmarkerRef.current;
          if (poseLandmarker) {
            try {
              const result = poseLandmarker.detectForVideo(
                video,
                performance.now(),
              );
              const picked = pickLargestLandmarks(result.landmarks);
              if (picked) {
                drawPose(ctx, picked.landmarks, canvas.width, canvas.height);
                const arms = extractPoseLandmarks(picked.landmarks);
                streamLandmarks = arms;
                const leftWristVis = arms.leftWrist?.visibility ?? 0;
                const rightWristVis = arms.rightWrist?.visibility ?? 0;
                notVisible =
                  leftWristVis < WRIST_VISIBILITY_THRESHOLD &&
                  rightWristVis < WRIST_VISIBILITY_THRESHOLD;
              } else {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
              }
            } catch {
              ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
          }
        }

        if (now - lastDebugUpdateRef.current > 100) {
          lastDebugUpdateRef.current = now;
          setDebugLandmarks(streamLandmarks);
          setLimbsNotVisible(Boolean(streamLandmarks && notVisible));
        }

        if (streamLandmarks) {
          const recorder = recorderRef.current;
          if (
            recorder.active &&
            now - recorder.lastCaptureAt >= RECORD_CAPTURE_INTERVAL_MS
          ) {
            recorder.lastCaptureAt = now;
            recorder.frames.push({
              timestamp: Date.now(),
              landmarks: streamLandmarks,
            });
          }

          if (now - lastWsSendRef.current >= WS_SEND_INTERVAL_MS) {
            lastWsSendRef.current = now;
            const ws = wsRef.current;
            if (ws && ws.readyState === WebSocket.OPEN) {
              const timestamp = Date.now();
              try {
                pendingSendAtRef.current.set(timestamp, performance.now());
                ws.send(
                  JSON.stringify({
                    mode: activeMode,
                    session_id: sessionIdRef.current || undefined,
                    landmarks: streamLandmarks,
                    timestamp,
                  }),
                );
                if (pendingSendAtRef.current.size > 50) {
                  const oldest = pendingSendAtRef.current.keys().next().value;
                  if (oldest !== undefined) {
                    pendingSendAtRef.current.delete(oldest);
                  }
                }
              } catch {
                // Send failures should not crash the detection loop.
              }
            }
          }
        }
      }

      rafRef.current = requestAnimationFrame(predict);
    };

    rafRef.current = requestAnimationFrame(predict);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [isStreaming, modelReady, mode]);

  useEffect(() => {
    if (!isStreaming) return;
    const video = videoRef.current;
    const stream = streamRef.current;
    if (!video || !stream) return;
    if (video.srcObject !== stream) video.srcObject = stream;
    void video.play().catch(() => {});
  }, [isStreaming]);

  useEffect(() => {
    let cancelled = false;

    fetch(`${API_BASE}/reference/status`)
      .then((response) => (response.ok ? response.json() : null))
      .then((data: ReferenceStatus | null) => {
        if (!cancelled && data) setReferenceStatus(data);
      })
      .catch(() => {});

    return () => {
      cancelled = true;
    };
  }, [calibrationOpen, rulebookOpen, trainOpen, isStreaming]);

  useEffect(() => {
    if (!isStreaming || mode !== "sign_language") {
      return;
    }
    let cancelled = false;

    fetch(`${API_BASE}/ml/stats`)
      .then((response) => (response.ok ? response.json() : null))
      .then((data: unknown) => {
        if (!cancelled && isSignTrainStats(data)) {
          setMlStats(data);
          setTrainError(null);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setTrainError("Could not load training stats.");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [trainOpen, isStreaming, mode]);

  function cancelCalibration() {
    calibrationTokenRef.current += 1;
    recorderRef.current.active = false;
    setCalibrationPhase("idle");
    setCalibrationGesture(null);
    setCalibrationMessage(null);
    setRecordProgress(0);
  }

  async function captureHold(
    gesture: CalibrationGesture,
  ): Promise<{ timestamp: number; landmarks: StreamLandmarks }[] | null> {
    const token = ++calibrationTokenRef.current;
    const isCurrent = () => calibrationTokenRef.current === token;
    const durationMs = RECORD_DURATION_MS[gesture];

    setCalibrationGesture(gesture);
    setCalibrationMessage(null);
    setCalibrationPhase("countdown");

    for (let value = 3; value >= 1; value -= 1) {
      if (!isCurrent()) return null;
      setCountdown(value);
      await sleep(1000);
    }
    if (!isCurrent()) return null;

    recorderRef.current = { active: true, frames: [], lastCaptureAt: 0 };
    setRecordProgress(0);
    setCalibrationPhase("recording");

    const startedAt = performance.now();
    for (;;) {
      if (!isCurrent()) {
        recorderRef.current.active = false;
        return null;
      }
      const elapsed = performance.now() - startedAt;
      setRecordProgress(Math.min(1, elapsed / durationMs));
      if (elapsed >= durationMs) break;
      await sleep(50);
    }

    recorderRef.current.active = false;
    const frames = recorderRef.current.frames;
    if (!isCurrent()) return null;

    if (frames.length < 3) {
      setCalibrationPhase("error");
      setCalibrationMessage(
        mode === "sign_language"
          ? "Not enough hand data captured — keep one hand clearly in frame."
          : "Not enough pose data captured — make sure your whole upper body is in frame.",
      );
      return null;
    }

    return frames;
  }

  async function runCalibration(gesture: CalibrationGesture) {
    const gestureType =
      gesture === "seatbelt_demo" || gesture === "wave" ? "motion" : "pose";
    const frames = await captureHold(gesture);
    if (!frames) return;

    setCalibrationPhase("saving");

    try {
      const response = await fetch(`${API_BASE}/reference/record`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          gesture,
          gesture_type: gestureType,
          landmark_sequence: frames,
        }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail ?? `Request failed (${response.status})`);
      }

      const saved = await response.json();

      if (saved?.reference_status) {
        setReferenceStatus(saved.reference_status as ReferenceStatus);
      }
      setCalibrationPhase("saved");
      setCalibrationMessage(`${GESTURE_LABELS[gesture]} reference saved ✓`);
    } catch (err) {
      setCalibrationPhase("error");
      setCalibrationMessage(
        err instanceof Error ? err.message : "Could not save reference.",
      );
    }
  }

  async function addTrainingSamples(gesture: SignGesture) {
    setTrainError(null);
    const frames = await captureHold(gesture);
    if (!frames) return;

    setCalibrationPhase("saving");

    try {
      const response = await fetch(`${API_BASE}/ml/samples`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          gesture,
          landmark_sequence: frames,
          session_id: sessionIdRef.current || undefined,
        }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail ?? `Request failed (${response.status})`);
      }

      const saved = (await response.json()) as {
        gold_holds?: number;
        stats?: unknown;
      };
      if (isSignTrainStats(saved.stats)) {
        setMlStats(saved.stats);
      }
      setCalibrationPhase("saved");
      const holds = saved.gold_holds ?? 0;
      const unit = gesture === "wave" ? "clips" : "holds";
      setCalibrationMessage(
        `${GESTURE_LABELS[gesture]}: ${holds} gold ${unit} saved. Talk still uses rules.`,
      );
    } catch (err) {
      setCalibrationPhase("error");
      const message =
        err instanceof Error ? err.message : "Could not save training sample.";
      setCalibrationMessage(message);
      setTrainError(message);
    }
  }

  async function readApiDetail(
    response: Response,
    fallback: string,
  ): Promise<string> {
    const body = (await response.json().catch(() => null)) as {
      detail?: unknown;
    } | null;
    if (typeof body?.detail === "string" && body.detail) {
      return body.detail;
    }
    return fallback;
  }

  async function trainSignModel() {
    setTrainError(null);
    setTrainBusy(true);
    try {
      const response = await fetch(`${API_BASE}/ml/train`, { method: "POST" });
      if (!response.ok) {
        throw new Error(
          await readApiDetail(response, "Training failed."),
        );
      }
      const saved = (await response.json()) as {
        model?: { val_accuracy?: number; version?: string };
        stats?: unknown;
      };
      if (isSignTrainStats(saved.stats)) {
        setMlStats(saved.stats);
      }
      const accuracy =
        typeof saved.model?.val_accuracy === "number"
          ? Math.round(saved.model.val_accuracy * 100)
          : null;
      setCalibrationMessage(
        accuracy !== null
          ? `Trained ${saved.model?.version ?? "model"} — test accuracy ${accuracy}%. Promote to use it in Talk.`
          : "Model trained. Promote to use it in Talk.",
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Training failed.";
      setTrainError(message);
    } finally {
      setTrainBusy(false);
    }
  }

  async function promoteSignModel() {
    setTrainError(null);
    setTrainBusy(true);
    try {
      const version = mlStats?.last_train?.version ?? mlStats?.model.version;
      const path = version
        ? `${API_BASE}/ml/models/${encodeURIComponent(version)}/promote`
        : `${API_BASE}/ml/runtime`;
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: version ? undefined : JSON.stringify({ source: "model" }),
      });
      if (!response.ok) {
        throw new Error(
          await readApiDetail(response, "Could not promote the model."),
        );
      }
      const saved = (await response.json()) as { stats?: unknown };
      if (isSignTrainStats(saved.stats)) {
        setMlStats(saved.stats);
      }
      setCalibrationMessage(
        "Promoted. Talk now uses the trained model for still signs.",
      );
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Could not promote the model.";
      setTrainError(message);
    } finally {
      setTrainBusy(false);
    }
  }

  async function useRulesAgain() {
    setTrainError(null);
    setTrainBusy(true);
    try {
      const response = await fetch(`${API_BASE}/ml/runtime`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: "heuristic" }),
      });
      if (!response.ok) {
        throw new Error(
          await readApiDetail(response, "Could not switch back to rules."),
        );
      }
      const saved = (await response.json()) as { stats?: unknown };
      if (isSignTrainStats(saved.stats)) {
        setMlStats(saved.stats);
      }
      setCalibrationMessage("Talk is using hardcoded rules again.");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Could not switch back to rules.";
      setTrainError(message);
    } finally {
      setTrainBusy(false);
    }
  }

  async function resetAllRecordedReferences() {
    setCalibrationMessage(null);
    try {
      const response = await fetch(`${API_BASE}/reference/reset`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error("Could not reset references.");
      }
      const data = (await response.json()) as {
        reference_status?: ReferenceStatus;
      };
      if (data.reference_status) {
        setReferenceStatus(data.reference_status);
      }
      setCalibrationMessage("Recorded references cleared. Using defaults.");
    } catch (err) {
      setCalibrationMessage(
        err instanceof Error ? err.message : "Could not reset references.",
      );
    }
  }

  async function resetOneRecordedReference(gesture: CalibrationGesture) {
    setCalibrationMessage(null);
    try {
      const response = await fetch(
        `${API_BASE}/reference/${encodeURIComponent(gesture)}`,
        { method: "DELETE" },
      );
      if (!response.ok) {
        throw new Error("Could not clear this reference.");
      }
      const data = (await response.json()) as {
        reference_status?: ReferenceStatus;
      };
      if (data.reference_status) {
        setReferenceStatus(data.reference_status);
      }
      setCalibrationMessage(
        `${GESTURE_LABELS[gesture]} is using the default again.`,
      );
    } catch (err) {
      setCalibrationMessage(
        err instanceof Error ? err.message : "Could not clear this reference.",
      );
    }
  }

  function pushTimelineEvent(event: {
    gesture: string;
    correct: boolean | null;
    score: number;
    coaching_text?: string | null;
  }) {
    if (recordingStartedAtRef.current === null) return;
    if (event.gesture === "none") return;
    const key = `${event.gesture}:${event.correct}:${Math.round(event.score * 20)}`;
    if (key === lastTimelineKeyRef.current) return;
    lastTimelineKeyRef.current = key;
    sessionTimelineRef.current.push({
      t_ms: Math.max(0, Date.now() - recordingStartedAtRef.current),
      ...event,
    });
  }

  function startSessionRecorder(stream: MediaStream) {
    mediaChunksRef.current = [];
    sessionTimelineRef.current = [];
    lastTimelineKeyRef.current = "";
    recordingStartedAtRef.current = Date.now();
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current = null;

    const mimeCandidates = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
    ];
    const mimeType = mimeCandidates.find((type) =>
      typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(type),
    );
    if (typeof MediaRecorder === "undefined") {
      console.warn("[recording] MediaRecorder unavailable");
      return;
    }
    try {
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) mediaChunksRef.current.push(ev.data);
      };
      recorder.start(1000);
      console.log("[recording] started", recorder.mimeType || mimeType);
    } catch (err) {
      console.warn("[recording] failed to start:", err);
      mediaRecorderRef.current = null;
    }
  }

  async function stopAndUploadSessionRecording() {
    const sessionId = sessionIdRef.current;
    const recorder = mediaRecorderRef.current;
    const startedAt = recordingStartedAtRef.current;
    mediaRecorderRef.current = null;
    recordingStartedAtRef.current = null;

    if (!recorder || !sessionId) return;

    const blob = await new Promise<Blob | null>((resolve) => {
      const finish = () => {
        const type = recorder.mimeType || "video/webm";
        const chunks = mediaChunksRef.current;
        mediaChunksRef.current = [];
        resolve(chunks.length ? new Blob(chunks, { type }) : null);
      };
      if (recorder.state === "inactive") {
        finish();
        return;
      }
      recorder.onstop = finish;
      try {
        recorder.stop();
      } catch {
        finish();
      }
    });

    if (!blob || blob.size < 1000) {
      console.warn("[recording] skip upload — empty blob");
      return;
    }

    const durationMs = startedAt ? Date.now() - startedAt : null;
    const form = new FormData();
    form.append("video", blob, `${sessionId}.webm`);
    form.append("timeline", JSON.stringify(sessionTimelineRef.current));
    form.append("mode", modeRef.current);
    if (durationMs !== null) form.append("duration_ms", String(durationMs));

    try {
      const response = await fetch(
        `${API_BASE}/sessions/${sessionId}/recording`,
        { method: "POST", body: form },
      );
      if (!response.ok) {
        console.warn("[recording] upload failed", response.status);
        return;
      }
      console.log("[recording] uploaded", sessionId, blob.size, "bytes");
    } catch (err) {
      console.warn("[recording] upload error", err);
    }
  }

  async function startWebcam() {
    setCameraError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = stream;
      sessionIdRef.current =
        typeof crypto !== "undefined" && crypto.randomUUID
          ? crypto.randomUUID().replace(/-/g, "")
          : `session-${Date.now()}`;
      if (modeRef.current === "aviation") {
        startSessionRecorder(stream);
      }
      setIsStreaming(true);
    } catch (err) {
      setCameraError(cameraPermissionMessage(err));
      setIsStreaming(false);
    }
  }

  async function stopWebcam() {
    cancelCalibration();
    setCalibrationOpen(false);
    setTrainOpen(false);
    stopAllSpeech();
    await stopAndUploadSessionRecording();

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) videoRef.current.srcObject = null;

    setIsStreaming(false);
  }

  const calibrationGestures: CalibrationGesture[] =
    mode === "aviation" ? AVIATION_GESTURES : SIGN_GESTURES;

  const modeLabel =
    mode === "aviation" ? "Aviation Training" : "Sign Language";
  const sessionSubtitle =
    mode === "aviation"
      ? gestureFeedback &&
        gestureFeedback.gesture !== "none" &&
        gestureFeedback.gesture in GESTURE_LABELS
        ? GESTURE_LABELS[gestureFeedback.gesture as CalibrationGesture]
        : "Exit Pointing"
      : composedSentence || lastSpokenSentence || "Talk with signs";
  const scorePercent =
    gestureFeedback && gestureFeedback.gesture !== "none"
      ? Math.round(Math.max(0, Math.min(1, gestureFeedback.score)) * 100)
      : 0;
  const calibrationBusy =
    calibrationPhase === "countdown" ||
    calibrationPhase === "recording" ||
    calibrationPhase === "saving";
  const liveTest = liveTestFromDebug(scoreDebug);

  if (isStreaming) {
    return (
      <div className="flex min-h-full flex-1 flex-col bg-background">
        <header className="flex flex-wrap items-center gap-3 border-b border-ink/10 bg-surface px-4 py-3 sm:gap-4 sm:px-6">
          <div className="min-w-0 flex-1 basis-[12rem]">
            <h1 className="font-heading truncate text-base font-medium text-ink sm:text-lg">
              {modeLabel}
              <span className="text-muted"> — {sessionSubtitle}</span>
            </h1>
          </div>
          <button
            type="button"
            onClick={() => setRulebookOpen((open) => !open)}
            className={`shrink-0 border px-3 py-1.5 text-xs transition-colors ${
              rulebookOpen
                ? "border-accent bg-accent text-white"
                : "border-ink/20 text-muted hover:border-ink/40 hover:text-ink"
            }`}
          >
            Signs
          </button>
          <button
            type="button"
            onClick={() => setCalibrationOpen((open) => !open)}
            className={`shrink-0 border px-3 py-1.5 text-xs transition-colors ${
              calibrationOpen
                ? "border-accent bg-accent text-white"
                : "border-ink/20 text-muted hover:border-ink/40 hover:text-ink"
            }`}
          >
            Record reference
          </button>
          {mode === "sign_language" && (
            <button
              type="button"
              onClick={() => setTrainOpen((open) => !open)}
              className={`shrink-0 border px-3 py-1.5 text-xs transition-colors ${
                trainOpen
                  ? "border-accent bg-accent text-white"
                  : "border-ink/20 text-muted hover:border-ink/40 hover:text-ink"
              }`}
            >
              Train model
            </button>
          )}
          <button
            type="button"
            onClick={stopWebcam}
            className="shrink-0 text-sm text-accent underline-offset-2 hover:underline"
          >
            Change mode
          </button>
          <Link
            href="/history"
            className="shrink-0 text-sm text-accent underline-offset-2 hover:underline"
          >
            History
          </Link>
          <div
            className={`flex shrink-0 items-center gap-2 border px-3 py-1.5 text-xs ${
              wsStatus === "connected"
                ? "border-success/30 bg-success/10 text-success"
                : wsStatus === "connecting" || wsStatus === "reconnecting"
                  ? "border-accent/30 bg-accent/10 text-accent"
                  : "border-alert/30 bg-alert/10 text-alert"
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                wsStatus === "connected"
                  ? "bg-success"
                  : wsStatus === "connecting" || wsStatus === "reconnecting"
                    ? "bg-accent"
                    : "bg-alert"
              }`}
              aria-hidden
            />
            {wsStatus === "connected"
              ? "Connected — streaming"
              : wsStatus === "connecting"
                ? "Connecting…"
                : wsStatus === "reconnecting"
                  ? "Disconnected — reconnecting…"
                  : "Disconnected"}
          </div>
        </header>

        <div className="relative flex flex-1 flex-col gap-4 p-4 lg:flex-row lg:items-start lg:gap-6 lg:p-6">
          {mode === "sign_language" && !rulebookOpen && (
            <button
              type="button"
              onClick={() => setRulebookOpen(true)}
              className="absolute left-0 top-28 z-20 border border-l-0 border-ink/20 bg-surface px-2 py-8 text-xs font-medium tracking-wide text-ink shadow-sm hover:bg-accent hover:text-white lg:top-24"
            >
              Signs
            </button>
          )}
          {mode === "sign_language" && (
            <SignRulebook
              open={rulebookOpen}
              selectedGesture={selectedRulebookGesture}
              referenceStatus={referenceStatus}
              recordBusy={calibrationBusy}
              onClose={() => setRulebookOpen(false)}
              onSelect={setSelectedRulebookGesture}
              onRecord={(gesture) => {
                setCalibrationOpen(true);
                void runCalibration(gesture);
              }}
              onResetOne={(gesture) => {
                void resetOneRecordedReference(gesture);
              }}
              onResetAll={() => {
                void resetAllRecordedReferences();
              }}
            />
          )}
          <div className="relative flex min-w-0 w-full flex-1 items-center justify-center">
            <div className="hud-frame relative aspect-video w-full max-w-[min(90vw,calc((100vh-7.5rem)*16/9))] max-h-[calc(100vh-7.5rem)] bg-ink lg:w-[min(85vw,calc((100vh-7.5rem)*16/9))]">
              <div className="hud-corners" aria-hidden />
              <div className="safe-zone" aria-hidden />
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="absolute inset-0 h-full w-full object-contain"
              />
              <canvas
                ref={canvasRef}
                className="pointer-events-none absolute inset-0 h-full w-full object-contain"
              />
              {limbsNotVisible && (
                <div className="pointer-events-none absolute left-1/2 top-4 z-10 -translate-x-1/2 border border-alert/40 bg-alert px-3 py-1.5 text-xs font-medium text-white shadow-sm">
                  {mode === "sign_language"
                    ? "Move closer — hand not fully visible"
                    : "Move back — arms not fully visible"}
                </div>
              )}

              {calibrationPhase === "countdown" && (
                <div className="pointer-events-none absolute inset-0 z-20 flex flex-col items-center justify-center bg-ink/45">
                  <span className="font-heading text-8xl font-medium text-white tabular-nums">
                    {countdown}
                  </span>
                  <span className="mt-2 text-sm text-white/80">
                    Get into position —{" "}
                    {calibrationGesture
                      ? GESTURE_LABELS[calibrationGesture]
                      : ""}
                  </span>
                </div>
              )}

              {calibrationPhase === "recording" && (
                <div className="pointer-events-none absolute inset-x-0 top-4 z-20 mx-auto w-64">
                  <div className="flex items-center justify-center gap-2 bg-alert px-3 py-1.5 text-xs font-medium text-white">
                    <span className="h-2 w-2 rounded-full bg-white" aria-hidden />
                    Recording…
                  </div>
                  <div className="h-1.5 w-full bg-white/30">
                    <div
                      className="h-full bg-white"
                      style={{ width: `${Math.round(recordProgress * 100)}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>

          <aside className="flex w-full shrink-0 flex-col gap-4 lg:w-96 xl:w-[26rem]">
            {mode === "sign_language" && (
              <SignTrainer
                open={trainOpen}
                stats={mlStats}
                selectedGesture={selectedTrainGesture}
                recordBusy={calibrationBusy}
                trainBusy={trainBusy}
                error={trainError}
                liveTest={liveTest}
                onClose={() => setTrainOpen(false)}
                onSelect={setSelectedTrainGesture}
                onAddSamples={(gesture) => {
                  setSelectedTrainGesture(gesture);
                  void addTrainingSamples(gesture);
                }}
                onTrain={() => {
                  void trainSignModel();
                }}
                onPromote={() => {
                  void promoteSignModel();
                }}
                onUseRules={() => {
                  void useRulesAgain();
                }}
              />
            )}
            {calibrationOpen && (
              <div className="border-2 border-accent/50 bg-surface p-5">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h2 className="font-heading text-lg font-medium text-ink">
                      Calibrate reference
                    </h2>
                    <p className="mt-1 text-xs leading-relaxed text-muted">
                      {mode === "sign_language"
                        ? "Record each sign with one hand clearly in frame."
                        : "Record each gesture yourself to replace the built-in defaults."}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      cancelCalibration();
                      setCalibrationOpen(false);
                    }}
                    className="shrink-0 text-xs text-muted hover:text-ink"
                  >
                    Close
                  </button>
                </div>

                <dl className="mt-4 max-h-48 space-y-1.5 overflow-y-auto border-t border-ink/10 pt-4 text-xs">
                  {calibrationGestures.map((gesture) => (
                    <div
                      key={gesture}
                      className="flex items-center justify-between gap-2"
                    >
                      <dt className="text-muted">
                        {GESTURE_LABELS[gesture]}
                        {mode === "sign_language" && (
                          <span className="text-ink/40">
                            {" "}
                            · {SIGN_MEANINGS[gesture as SignGesture]}
                          </span>
                        )}
                      </dt>
                      <dd
                        className={
                          referenceStatus?.[gesture] === "recorded"
                            ? "shrink-0 text-success"
                            : "shrink-0 text-muted"
                        }
                      >
                        {referenceStatus
                          ? referenceStatus[gesture] === "recorded"
                            ? "Recorded"
                            : "Using default"
                          : "…"}
                      </dd>
                    </div>
                  ))}
                </dl>

                <div className="mt-4 flex max-h-56 flex-col gap-2 overflow-y-auto">
                  {calibrationGestures.map((gesture) => (
                    <button
                      key={gesture}
                      type="button"
                      onClick={() => runCalibration(gesture)}
                      disabled={calibrationBusy}
                      className="border border-ink/20 px-3 py-2 text-left text-sm text-ink transition-colors hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      Record {GESTURE_LABELS[gesture]}
                      <span className="ml-1 text-xs text-muted">
                        ({RECORD_DURATION_MS[gesture] / 1000}s)
                      </span>
                    </button>
                  ))}
                </div>

                <button
                  type="button"
                  onClick={() => void resetAllRecordedReferences()}
                  disabled={calibrationBusy}
                  className="mt-3 min-h-11 w-full border border-ink/20 px-3 py-2 text-sm text-ink disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Clear all recorded references
                </button>

                {calibrationPhase === "recording" && (
                  <div className="mt-4">
                    <div className="mb-1.5 flex items-center justify-between text-xs">
                      <span className="text-alert">Recording…</span>
                      <span className="text-muted tabular-nums">
                        {Math.round(recordProgress * 100)}%
                      </span>
                    </div>
                    <div className="h-1.5 w-full bg-ink/10">
                      <div
                        className="h-full bg-alert"
                        style={{
                          width: `${Math.round(recordProgress * 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                )}

                {calibrationPhase === "countdown" && (
                  <p className="mt-4 text-xs text-muted">
                    Starting in {countdown}…
                  </p>
                )}
                {calibrationPhase === "saving" && (
                  <p className="mt-4 text-xs text-muted">Saving reference…</p>
                )}
                {calibrationPhase === "saved" && calibrationMessage && (
                  <p className="mt-4 text-xs text-success">
                    {calibrationMessage} Try the gesture now.
                  </p>
                )}
                {calibrationPhase === "error" && calibrationMessage && (
                  <p className="mt-4 text-xs text-alert">{calibrationMessage}</p>
                )}
              </div>
            )}

            {mode === "sign_language" ? (
              <SignCommunicator
                voiceUnlocked={voiceUnlocked}
                speaking={speaking}
                tokens={sentenceTokens}
                sentence={composedSentence}
                lastSpoken={lastSpokenSentence}
                currentMeaning={
                  gestureFeedback?.correct ? (gestureFeedback.meaning ?? null) : null
                }
                currentLabel={
                  gestureFeedback &&
                  gestureFeedback.gesture !== "none" &&
                  gestureFeedback.gesture in GESTURE_LABELS
                    ? GESTURE_LABELS[gestureFeedback.gesture as CalibrationGesture]
                    : null
                }
                recognized={Boolean(gestureFeedback?.correct)}
                scorePercent={scorePercent}
                motionProgress={motionProgress}
                motionHint={motionHint}
                handStyle={handStyle}
                onEnableVoice={() => {
                  setVoiceUnlocked(true);
                  if (typeof window !== "undefined" && "speechSynthesis" in window) {
                    window.speechSynthesis.getVoices();
                  }
                  unlockSpeechSynthesis({
                    onStart: () => setSpeaking(true),
                    onEnd: () => setSpeaking(false),
                  });
                }}
                onSpeak={() => speakComposedSentence(composedSentence)}
                onRepeat={() => {
                  if (!lastSpokenSentence) return;
                  speakSentence(lastSpokenSentence, {
                    onStart: () => setSpeaking(true),
                    onEnd: () => setSpeaking(false),
                  });
                }}
                onClear={resetSentenceBuilder}
              />
            ) : (
            <div className="border border-ink/15 bg-surface p-5">
              <p className="text-xs text-muted">Current result</p>
              {!gestureFeedback ? (
                <p className="mt-3 text-sm leading-relaxed text-muted">
                  Move into Exit Pointing or Seatbelt Demo. Feedback appears
                  while you are actively training — not while standing still.
                </p>
              ) : gestureFeedback.gesture === "none" ? (
                <div className="mt-3">
                  <h2 className="font-heading text-2xl font-medium text-muted">
                    Ready when you move
                  </h2>
                  <p className="mt-2 text-sm text-muted">
                    Raise both arms for Exit Pointing, or slide both hands
                    together at your waist for Seatbelt Demo.
                  </p>
                </div>
              ) : (
                <>
                  <h2 className="font-heading mt-2 text-2xl font-medium text-ink">
                    {
                      GESTURE_LABELS[
                        gestureFeedback.gesture as CalibrationGesture
                      ]
                    }
                  </h2>
                  <div
                    className={`mt-4 inline-block px-4 py-2 text-sm font-medium text-white ${
                      gestureFeedback.correct ? "bg-success" : "bg-alert"
                    }`}
                  >
                    {gestureFeedback.correct
                      ? "Correct — keep going"
                      : "You're doing it wrong"}
                  </div>
                  <p className="mt-3 text-sm text-muted">
                    {gestureFeedback.correct
                      ? gestureFeedback.gesture === "seatbelt_demo"
                        ? "Finish the waist slide smoothly."
                        : "Hold the line — elbows nearly straight, wrists level."
                      : coachingText ||
                        gestureFeedback.hints?.[0] ||
                        "Adjust toward the target zone."}
                  </p>
                  <div className="mt-5">
                    <div className="mb-2 flex items-center justify-between text-sm">
                      <span className="text-muted">Confidence</span>
                      <span className="font-medium text-ink">
                        {scorePercent}%
                      </span>
                    </div>
                    <div className="h-2 w-full bg-ink/10">
                      <div
                        className={`h-full transition-[width] duration-300 ${
                          gestureFeedback.correct ? "bg-success" : "bg-alert"
                        }`}
                        style={{ width: `${scorePercent}%` }}
                      />
                    </div>
                  </div>
                  {gestureFeedback.gesture === "exit_pointing" && (
                    <dl className="mt-5 grid grid-cols-2 gap-x-3 gap-y-2 text-xs text-muted">
                      <div>
                        <dt>Left arm</dt>
                        <dd className="font-mono text-ink">
                          {gestureFeedback.left_arm_angle ?? "—"}°
                        </dd>
                      </div>
                      <div>
                        <dt>Right arm</dt>
                        <dd className="font-mono text-ink">
                          {gestureFeedback.right_arm_angle ?? "—"}°
                        </dd>
                      </div>
                      <div>
                        <dt>Target</dt>
                        <dd className="font-mono text-ink">
                          {gestureFeedback.target_angle ?? "—"}°
                        </dd>
                      </div>
                      <div>
                        <dt>Comfort zone</dt>
                        <dd className="font-mono text-ink">
                          ±
                          {gestureFeedback.pass_band_deg ??
                            gestureFeedback.tolerance ??
                            "—"}
                          °
                        </dd>
                      </div>
                    </dl>
                  )}

                  {!gestureFeedback.correct &&
                    (coachingLoading || coachingText) && (
                      <div className="mt-5 border border-accent/30 bg-accent/5 px-4 py-3">
                        <p className="text-xs font-medium text-accent">
                          Coaching
                        </p>
                        {coachingLoading && !coachingText ? (
                          <p className="mt-1.5 text-sm text-muted">
                            Getting feedback…
                          </p>
                        ) : (
                          <p className="mt-1.5 text-sm leading-relaxed text-ink">
                            {coachingText}
                          </p>
                        )}
                      </div>
                    )}
                </>
              )}
            </div>
            )}

            <div className="border border-ink/15 bg-surface">
              <button
                type="button"
                onClick={() => setDebugOpen((open) => !open)}
                className="flex w-full items-center justify-between px-4 py-3 text-left text-sm text-ink transition-colors hover:bg-ink/5"
              >
                <span>Debug data</span>
                <span className="text-muted">{debugOpen ? "Hide" : "Show"}</span>
              </button>
              {debugOpen && (
                <div className="border-t border-ink/10 px-4 py-3">
                  <p className="mb-3 text-xs text-muted">
                    Last round-trip time:{" "}
                    {lastRoundTripMs !== null ? `${lastRoundTripMs} ms` : "—"}
                  </p>
                  <p className="mb-2 text-xs font-medium text-ink">
                    Scoring response
                  </p>
                  <pre className="mb-4 max-h-40 overflow-auto font-mono text-xs leading-relaxed text-muted">
                    {scoreDebug
                      ? JSON.stringify(scoreDebug, null, 2)
                      : "Waiting for scoring response…"}
                  </pre>
                  {scoreDebug &&
                    typeof scoreDebug.scores === "object" &&
                    scoreDebug.scores !== null && (
                      <>
                        <p className="mb-2 text-xs font-medium text-ink">
                          Per-sign scores
                        </p>
                        <pre className="mb-4 max-h-40 overflow-auto font-mono text-xs leading-relaxed text-muted">
                          {JSON.stringify(scoreDebug.scores, null, 2)}
                        </pre>
                      </>
                    )}
                  <p className="mb-2 text-xs font-medium text-ink">Landmarks</p>
                  <pre className="max-h-48 overflow-auto font-mono text-xs leading-relaxed text-muted">
                    {debugLandmarks
                      ? JSON.stringify(debugLandmarks, null, 2)
                      : "Waiting for landmarks…"}
                  </pre>
                </div>
              )}
            </div>
          </aside>
        </div>
      </div>
    );
  }

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col items-center gap-14 px-6 pb-20 pt-24">
      <div className="flex w-full justify-end">
        <Link
          href="/history"
          className="text-sm text-accent underline-offset-2 hover:underline"
        >
          History
        </Link>
      </div>
      <h1 className="font-heading text-center text-4xl font-medium tracking-tight text-ink sm:text-[2.75rem]">
        AI Gesture Recognition Platform
      </h1>

      <section className="flex w-full flex-col gap-5">
        <p className="text-center text-sm text-muted">
          Choose how you want to practice
        </p>
        <div className="grid w-full gap-4 sm:grid-cols-2">
          <button
            type="button"
            onClick={() => setMode("aviation")}
            className={`border-2 bg-surface px-6 py-7 text-left transition-colors ${
              mode === "aviation"
                ? "border-accent"
                : "border-ink/15 hover:border-ink/30"
            }`}
          >
            <span className="font-heading block text-lg font-medium text-ink">
              Aviation Training
            </span>
            <span className="mt-2 block text-sm leading-relaxed text-muted">
              Practice standard hand signals used on the flight deck and ramp.
            </span>
          </button>
          <button
            type="button"
            onClick={() => setMode("sign_language")}
            className={`border-2 bg-surface px-6 py-7 text-left transition-colors ${
              mode === "sign_language"
                ? "border-accent"
                : "border-ink/15 hover:border-ink/30"
            }`}
          >
            <span className="font-heading block text-lg font-medium text-ink">
              Sign Language
            </span>
            <span className="mt-2 block text-sm leading-relaxed text-muted">
              Show signs one after another. They become a sentence, then the
              computer speaks it out loud.
            </span>
          </button>
        </div>
        <p className="text-center text-sm text-muted">
          Active mode: {modeLabel}
        </p>
      </section>

      <section className="flex w-full flex-col items-center gap-6">
        <button
          type="button"
          onClick={startWebcam}
          className="bg-accent px-6 py-3 text-sm font-medium text-white transition-colors hover:brightness-95"
        >
          Enable Webcam
        </button>

        {!poseReady && !handReady && !modelError && (
          <p className="text-sm text-muted">Loading detection models…</p>
        )}
        {modelError && <p className="text-sm text-alert">{modelError}</p>}
        {cameraError && (
          <div className="max-w-md border border-alert/30 bg-alert/5 px-4 py-3 text-center text-sm text-alert">
            {cameraError}
          </div>
        )}
      </section>
    </main>
  );
}
