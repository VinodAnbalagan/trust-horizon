// All data is precomputed by site/scripts/export_site_data.py and bundled: the site makes no network calls.
import siteJson from "./data/site.json";
import dreamsJson from "./data/dreams.json";

export interface Dream {
  split: "seen" | "unseen";
  T: number;
  onset: number | null;          // first broken step (primary label), null if it never broke
  err: number[];                 // latent error / the task's chance error
  u_r: number[];
  u_f: number[];
  u_s: number[];                 // signals divided by the task's mean latent motion
  p_forecast: number[];          // model: P(breaks within the next 8 steps)
  p_detect: number[];            // model: P(already broken)
  dpsnr: number[];
  median_horizon: number | null; // Kaplan-Meier median for the task, null if above the horizon
}

export const dreams = dreamsJson as unknown as Record<string, Dream>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const site = siteJson as any;

const frameUrls = import.meta.glob("./assets/frames/*.webp", { eager: true, query: "?url", import: "default" }) as Record<string, string>;
export function frameUrl(task: string, kind: "real" | "dream"): string {
  return frameUrls[`./assets/frames/${task}_${kind}.webp`];
}

export const seenTasks = Object.keys(dreams).filter((k) => dreams[k].split === "seen");
export const unseenTasks = Object.keys(dreams).filter((k) => dreams[k].split === "unseen");

export function pretty(task: string): string {
  return task
    .replace(/^pygame-/, "")
    .replace(/^atari-/, "Atari ")
    .replace(/^og-/, "OGBench ")
    .replace(/^ms-/, "ManiSkill ")
    .replace(/^mw-/, "MetaWorld ")
    .replace(/^rd-/, "RoboDesk ")
    .replace(/^mujoco-/, "MuJoCo ")
    .replace(/-/g, " ");
}

// One cache of decoded sprite sheets, shared by every canvas.
const cache = new Map<string, Promise<HTMLImageElement>>();
export function loadSprite(url: string): Promise<HTMLImageElement> {
  if (!cache.has(url)) {
    cache.set(url, new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = rej;
      im.src = url;
    }));
  }
  return cache.get(url)!;
}

/** Draw dream step t (1-based) from a 12 x 8 sprite of 64 px frames. */
export function drawFrame(ctx: CanvasRenderingContext2D, im: HTMLImageElement, t: number) {
  const i = t - 1, c = i % 12, r = Math.floor(i / 12);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(im, c * 64, r * 64, 64, 64, 0, 0, ctx.canvas.width, ctx.canvas.height);
}
