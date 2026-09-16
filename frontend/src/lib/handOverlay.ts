import {
  HandLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

export type TrailPoint = { x: number; y: number; t: number };
export type HandDrawStyle = "motion" | "hold";

const ACCENT = "#2C6E8C";
const HOLD = "#2F9E6E";
const TIP_INDEXES = [4, 8, 12, 16, 20] as const;
const MAX_INDEX_TRAIL = 24;
const MAX_TIP_TRAIL = 120;
export const TRAIL_MS = 1000;
const MOTION_X_STD = 0.04;

export function pushIndexTrail(
  trail: TrailPoint[],
  landmarks: NormalizedLandmark[],
  now: number,
): TrailPoint[] {
  const indexTip = landmarks[8];
  if (!indexTip) {
    return trail.filter((point) => now - point.t <= TRAIL_MS);
  }
  return [...trail, { x: indexTip.x, y: indexTip.y, t: now }]
    .filter((point) => now - point.t <= TRAIL_MS)
    .slice(-MAX_INDEX_TRAIL);
}

export function pushFingertipTrail(
  trail: TrailPoint[],
  landmarks: NormalizedLandmark[],
  now: number,
): TrailPoint[] {
  const added: TrailPoint[] = [];
  for (const index of TIP_INDEXES) {
    const tip = landmarks[index];
    if (!tip) continue;
    added.push({ x: tip.x, y: tip.y, t: now });
  }
  return [...trail, ...added]
    .filter((point) => now - point.t <= TRAIL_MS)
    .slice(-MAX_TIP_TRAIL);
}

export function trailMotionStd(trail: TrailPoint[]): number {
  if (trail.length < 5) return 0;
  const recent = trail.slice(-12);
  const xs = recent.map((point) => point.x);
  const mean = xs.reduce((sum, value) => sum + value, 0) / xs.length;
  const variance =
    xs.reduce((sum, value) => sum + (value - mean) ** 2, 0) / xs.length;
  return Math.sqrt(variance);
}

export function isTrailMoving(trail: TrailPoint[]): boolean {
  return trailMotionStd(trail) >= MOTION_X_STD;
}

function drawTrail(
  ctx: CanvasRenderingContext2D,
  trail: TrailPoint[],
  width: number,
  height: number,
  now: number,
) {
  trail.forEach((point, index) => {
    const age = now - point.t;
    const life = Math.max(0, 1 - age / TRAIL_MS);
    const radius = 2.5 + (index / Math.max(1, trail.length)) * 5;
    ctx.fillStyle = `rgba(44, 110, 140, ${0.15 + life * 0.7})`;
    ctx.beginPath();
    ctx.arc(point.x * width, point.y * height, radius, 0, Math.PI * 2);
    ctx.fill();
  });
}

export function drawHandOverlay(
  ctx: CanvasRenderingContext2D,
  hands: NormalizedLandmark[][],
  trail: TrailPoint[],
  width: number,
  height: number,
  style: HandDrawStyle,
) {
  ctx.clearRect(0, 0, width, height);
  const now =
    typeof performance !== "undefined" ? performance.now() : Date.now();

  if (style === "motion" || trail.length > 3) {
    drawTrail(ctx, trail, width, height, now);
  }

  const moving = style === "motion";
  ctx.strokeStyle = moving ? ACCENT : HOLD;
  ctx.fillStyle = moving ? ACCENT : HOLD;
  ctx.lineWidth = moving ? 1.25 : 3.25;
  ctx.lineCap = "round";
  ctx.globalAlpha = moving ? 0.4 : 1;

  for (const landmarks of hands) {
    for (const connection of HandLandmarker.HAND_CONNECTIONS) {
      const start = landmarks[connection.start];
      const end = landmarks[connection.end];
      if (!start || !end) continue;
      ctx.beginPath();
      ctx.moveTo(start.x * width, start.y * height);
      ctx.lineTo(end.x * width, end.y * height);
      ctx.stroke();
    }

    landmarks.forEach((landmark, index) => {
      const isTip = (TIP_INDEXES as readonly number[]).includes(index);
      const radius = moving ? (isTip ? 3.2 : 2) : isTip ? 6 : 3.8;
      ctx.beginPath();
      ctx.arc(landmark.x * width, landmark.y * height, radius, 0, Math.PI * 2);
      ctx.fill();
    });

    if (!moving) {
      const wrist = landmarks[0];
      const middleMcp = landmarks[9];
      if (wrist && middleMcp) {
        const cx = ((wrist.x + middleMcp.x) / 2) * width;
        const cy = ((wrist.y + middleMcp.y) / 2) * height;
        ctx.globalAlpha = 0.9;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, 18, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }

  ctx.globalAlpha = 1;
}
