import { useEffect, useRef, useState } from "react";
import { dreams, frameUrl, loadSprite, pretty } from "../data";
import { Thumb } from "./Figures";

const reduceMotion = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/** A real frame, optionally covered in noise (0..1). */
function Tile({ task, kind, step, noise = 0, size = 96, outline }: { task: string; kind: "real" | "dream"; step: number; noise?: number; size?: number; outline?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let live = true;
    loadSprite(frameUrl(task, kind)).then((im) => {
      const c = ref.current?.getContext("2d"); if (!live || !c) return;
      const i = step - 1, cc = i % 12, rr = Math.floor(i / 12);
      c.imageSmoothingEnabled = false;
      c.clearRect(0, 0, 64, 64);
      c.drawImage(im, cc * 64, rr * 64, 64, 64, 0, 0, 64, 64);
      if (noise > 0) {
        const px = c.getImageData(0, 0, 64, 64);
        let seed = step * 7919 + Math.round(noise * 1000);
        const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
        for (let k = 0; k < px.data.length; k += 4) {
          const n = rnd() * 255;
          for (let ch = 0; ch < 3; ch++) px.data[k + ch] = px.data[k + ch] * (1 - noise) + n * noise;
        }
        c.putImageData(px, 0, 0);
      }
    });
    return () => { live = false; };
  }, [task, kind, step, noise]);
  return <canvas ref={ref} width={64} height={64} style={{ width: size, height: size, imageRendering: "pixelated", borderRadius: 3, boxShadow: outline ? `0 0 0 2px ${outline}` : undefined, display: "block" }} />;
}

/** The compact code of a frame, drawn as a small grid; `changed` cells are outlined. */
function Code({ seed, changed, highlight }: { seed: number; changed: number[]; highlight?: boolean }) {
  const cells = Array.from({ length: 24 }, (_, k) => {
    let x = (seed * 9301 + k * 49297) % 233280; x = x / 233280;
    const changedHere = changed.includes(k);
    const v = changedHere ? (x + 0.5) % 1 : x;
    return { v, changedHere };
  });
  return (
    <div className={`code${highlight ? " lit" : ""}`}>
      {cells.map((c, k) => (
        <i key={k} style={{ background: `color-mix(in srgb, var(--dream) ${Math.round(20 + c.v * 70)}%, var(--paper-2))`, outline: c.changedHere && highlight ? "2px solid var(--broken)" : undefined }} />
      ))}
    </div>
  );
}

function Meter({ value, label }: { value: number; label: string }) {
  return (
    <div className="meter">
      <div className="meter-bar"><i style={{ width: `${Math.round(value * 100)}%`, background: value > 0.5 ? "var(--broken)" : "var(--real)" }} /></div>
      <span>{label}</span>
    </div>
  );
}

// "Sure": early in a dream that is still close to reality. "Unsure": a pong dream that has already broken.
const SURE = { task: "cheetah-run", step: 5 };
const UNSURE = { task: "pygame-pong", step: 20 };

export function SignalsExplainer() {
  const [mode, setMode] = useState<"sure" | "unsure">("sure");
  const [phase, setPhase] = useState(reduceMotion ? 4 : 0);
  useEffect(() => {
    if (reduceMotion) return;
    setPhase(0);
    const id = window.setInterval(() => setPhase((p) => (p + 1) % 7), 650);
    return () => window.clearInterval(id);
  }, [mode]);
  const p = Math.min(phase, 4);
  const m = mode === "sure" ? SURE : UNSURE;
  const unsure = mode === "unsure";
  const changed = unsure ? [1, 4, 7, 9, 14, 17, 20, 22] : [11];
  // second thoughts: four cleaning steps; when unsure the guess keeps switching
  const cleanSteps = [0.75, 0.5, 0.25, 0];
  const guessStep = (k: number) => (unsure ? m.step + [0, 9, 3, 14][k] : m.step);

  return (
    <figure className="fig explainer">
      <div className="explainer-head">
        <p className="chart-title" style={{ margin: 0 }}>Show a moment where the world model is</p>
        <div className="seg" role="group" aria-label="Which moment">
          <button aria-pressed={!unsure} onClick={() => setMode("sure")}>sure</button>
          <button aria-pressed={unsure} onClick={() => setMode("unsure")}>unsure</button>
        </div>
        <span className="muted">{unsure ? `${pretty(m.task)}, step ${m.step}: this dream broke at step 4` : `${pretty(m.task)}, step ${m.step}: this dream is still close to reality`}</span>
      </div>

      <div className="sig3">
        <div className="sig ur">
          <h4>Round trip <span>u<sub>r</sub></span></h4>
          <p className="q">Turn the imagined code into a picture and back. Does the code come back the same?</p>
          <div className="rt">
            <Code seed={m.step * 13 + (unsure ? 5 : 0)} changed={[]} highlight={p >= 0} />
            <span className={`arr${p >= 1 ? " on" : ""}`}>→ picture →</span>
            <div style={{ opacity: p >= 1 ? 1 : 0.15, transition: "opacity 200ms" }}><Tile task={m.task} kind="dream" step={m.step} size={64} /></div>
            <span className={`arr${p >= 2 ? " on" : ""}`}>→ code →</span>
            <div style={{ opacity: p >= 2 ? 1 : 0.15, transition: "opacity 200ms" }}><Code seed={m.step * 13 + (unsure ? 5 : 0)} changed={changed} highlight={p >= 3} /></div>
          </div>
          <Meter value={p >= 3 ? (unsure ? 0.8 : 0.08) : 0} label={p >= 3 ? (unsure ? "many cells changed: unsure" : "almost nothing changed: sure") : "comparing..."} />
          <p className="everyday">Like translating a sentence to French and back: a clear sentence survives unchanged.</p>
        </div>

        <div className="sig uf">
          <h4>Second thoughts <span>u<sub>f</sub></span></h4>
          <p className="q">The world model makes a frame by cleaning up noise in a few steps. Does its guess keep changing?</p>
          <div className="steps4">
            {cleanSteps.map((nz, k) => (
              <div key={k} style={{ opacity: p >= k ? 1 : 0.12, transition: "opacity 200ms" }}>
                <Tile task={m.task} kind="dream" step={Math.min(guessStep(k), dreams[m.task].T)} noise={nz} size={56} />
                <span>clean {k + 1}</span>
              </div>
            ))}
          </div>
          <Meter value={p >= 4 ? (unsure ? 0.75 : 0.1) : (p / 4) * (unsure ? 0.75 : 0.1)} label={p >= 4 ? (unsure ? "the guess kept changing: unsure" : "the same picture sharpened: sure") : "cleaning..."} />
          <p className="everyday">Like a student who keeps rewriting an answer, against one who writes it once and only neatens it.</p>
        </div>

        <div className="sig us">
          <h4>Two tries disagree <span>u<sub>s</sub></span></h4>
          <p className="q">Imagine the same next frame twice, from two different random starts. Do the two tries agree?</p>
          <div className="tries">
            <div style={{ opacity: p >= 1 ? 1 : 0.12, transition: "opacity 200ms" }}><span>try 1</span><Tile task={m.task} kind="dream" step={m.step} size={80} /></div>
            <div style={{ opacity: p >= 2 ? 1 : 0.12, transition: "opacity 200ms" }}><span>try 2</span><Tile task={m.task} kind="dream" step={unsure ? m.step + 11 : m.step} size={80} outline={unsure && p >= 3 ? "var(--broken)" : undefined} /></div>
          </div>
          <Meter value={p >= 3 ? (unsure ? 0.7 : 0.05) : 0} label={p >= 3 ? (unsure ? "the tries disagree: unsure" : "the tries match: sure") : "imagining twice..."} />
          <p className="everyday">Like asking two witnesses: if their stories differ, trust neither.</p>
        </div>
      </div>
      <figcaption className="note">
        An illustration built from real frames: the steps are drawn to show the idea, not measured here. The real measurements for every moment are in the table below. Higher means less sure.
      </figcaption>
    </figure>
  );
}

/* ======================================================================== the data table */

const ROWS = [
  { task: "cheetah-run", step: 5 },
  { task: "cheetah-run", step: 33 },
  { task: "cheetah-run", step: 50 },
  { task: "pygame-pong", step: 2 },
  { task: "pygame-point-maze-var1", step: 40 },
  { task: "walker-walk", step: 70 },
];

const IconRT = () => <svg viewBox="0 0 28 16" width="28" height="16" aria-hidden="true"><path d="M5 8a5 5 0 0 1 9-3M23 8a5 5 0 0 1-9 3" fill="none" stroke="var(--s-ur)" strokeWidth="1.8" /><path d="M14 2v3h-3M14 14v-3h3" fill="none" stroke="var(--s-ur)" strokeWidth="1.8" /></svg>;
const IconUF = () => <svg viewBox="0 0 28 16" width="28" height="16" aria-hidden="true"><path d="M2 12 L7 4 L11 12 L16 3 L20 11 L26 6" fill="none" stroke="var(--s-uf)" strokeWidth="1.8" strokeLinejoin="round" /></svg>;
const IconUS = () => <svg viewBox="0 0 28 16" width="28" height="16" aria-hidden="true"><rect x="3" y="3" width="10" height="10" rx="2" fill="none" stroke="var(--s-us)" strokeWidth="1.8" /><rect x="15" y="3" width="10" height="10" rx="2" fill="none" stroke="var(--s-us)" strokeWidth="1.8" strokeDasharray="3 2" /></svg>;
const IconStep = () => <svg viewBox="0 0 28 16" width="28" height="16" aria-hidden="true"><path d="M3 13 H25 M3 13 V9 M10 13 V9 M17 13 V9 M24 13 V5" fill="none" stroke="var(--ink-2)" strokeWidth="1.8" /></svg>;
const IconBroken = () => <svg viewBox="0 0 28 16" width="28" height="16" aria-hidden="true"><path d="M3 8 H25" stroke="var(--real)" strokeWidth="2.2" strokeDasharray="5 3" /></svg>;
const IconSoon = () => <svg viewBox="0 0 28 16" width="28" height="16" aria-hidden="true"><path d="M3 8 H25" stroke="var(--dream)" strokeWidth="2.2" /></svg>;

export function DataTable() {
  const fmt = (v: number) => (v >= 1 ? v.toFixed(2) : v >= 0.1 ? v.toFixed(2) : v.toFixed(3));
  return (
    <figure className="fig">
      <p className="lede" style={{ marginBottom: "0.6rem" }}><b>160 test dreams, 14,600 moments</b>, each labelled broken or not. Here are 6 of them:</p>
      <div className="tablewrap">
        <table className="datatable">
          <thead>
            <tr className="xy"><th></th><th>x<sub>1</sub></th><th>x<sub>2</sub></th><th>x<sub>3</sub></th><th>x<sub>4</sub></th><th className="ysep">y<sub>1</sub></th><th>y<sub>2</sub></th></tr>
            <tr className="icons"><th></th><th><IconRT /></th><th><IconUF /></th><th><IconUS /></th><th><IconStep /></th><th className="ysep"><IconBroken /></th><th><IconSoon /></th></tr>
            <tr>
              <th>moment (real | imagined)</th>
              <th>round trip</th><th>second thoughts</th><th>two tries disagree</th><th>step</th>
              <th className="ysep">broken now?</th><th>breaks within 8?</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map(({ task, step }) => {
              const d = dreams[task], i = step - 1;
              const brokenNow = d.onset != null && step >= d.onset;
              const soon = !brokenNow && d.onset != null && d.onset - step <= 8 ? 1 : 0;
              return (
                <tr key={task + step} className={task === "pygame-point-maze-var1" ? "callout" : undefined}>
                  <td>
                    <div className="moment">
                      <div className="mini"><Thumb task={task} kind="real" step={step} /><Thumb task={task} kind="dream" step={step} /></div>
                      <span>{pretty(task)}{task === "pygame-point-maze-var1" ? <b className="star"> *</b> : null}<br /><span className="muted">step {step}</span></span>
                    </div>
                  </td>
                  <td>{fmt(d.u_r[i])}</td><td>{fmt(d.u_f[i])}</td><td>{fmt(d.u_s[i])}</td><td>{step}</td>
                  <td className="ysep y alarm">{brokenNow ? 1 : 0}</td>
                  <td className="y">{brokenNow ? <span className="muted" title="already broken">n/a</span> : soon}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="muted" style={{ fontStyle: "italic", fontSize: "var(--t-small)", margin: "0.2rem 0 1.25rem" }}>… and 14,594 more moments</p>

      <div className="xy-legend">
        <p className="xlab">x: what the predictions see</p>
        <p className="ylab">y: what they try to guess (1 yes, 0 no)</p>
      </div>
      <div className="defs">
        <div><IconRT /><p><b>round trip</b>: how much the imagined code changes when turned into a picture and back. <span className="muted">Higher means less sure.</span></p></div>
        <div><IconBroken /><p><b>broken now?</b>: 1 if the dream has broken by this step. <span className="muted">This is what the Alarm guesses.</span></p></div>
        <div><IconUF /><p><b>second thoughts</b>: how much the guess changes while the noise is cleaned up. <span className="muted">Higher means less sure.</span></p></div>
        <div><IconSoon /><p><b>breaks within 8?</b>: 1 if a dream that is still fine breaks in the next 8 steps. <span className="muted">This is what the Forecast guesses; n/a once broken.</span></p></div>
        <div><IconUS /><p><b>two tries disagree</b>: how far apart two imagined versions of the same frame are. <span className="muted">Higher means less sure.</span></p></div>
        <div><IconStep /><p><b>step</b>: how far into the dream. <span className="muted">Late steps break more often, whatever the signals say.</span></p></div>
      </div>
      <figcaption className="note">
        <b className="star">*</b> Look at the point maze row: its signals are the highest in the table, yet that dream never breaks. Each task has its own normal level, so a number means little until you compare it with other moments of the same task. That is why the results below compare moments within one task. The predictions actually see 11 numbers per moment: for each signal its current value, its average over the last 4 steps and its trend, plus scene motion and the step. Signals are divided by how much the task's scene usually moves.
      </figcaption>
    </figure>
  );
}
