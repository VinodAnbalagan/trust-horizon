import { useEffect, useRef, useState } from "react";
import { Thumb } from "./Figures";
import LineChart from "./LineChart";
import { dreams, pretty } from "../data";

const reduceMotion = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/** Steps 1..T on a timer, with a pause at the end before looping. */
function useTicker(T: number, ms: number, holdEnd = 6, startAt = 1) {
  const [t, setT] = useState(reduceMotion ? T : startAt);
  const [playing, setPlaying] = useState(!reduceMotion);
  const hold = useRef(0);
  useEffect(() => { setT(reduceMotion ? T : startAt); hold.current = 0; }, [T, startAt]);
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setT((x) => {
        if (x < T) return x + 1;
        if (hold.current++ < holdEnd) return x;
        hold.current = 0;
        return startAt;
      });
    }, ms);
    return () => window.clearInterval(id);
  }, [playing, T, ms, holdEnd, startAt]);
  return { t, setT, playing, setPlaying };
}

/* ======================================================================== 1. the dream machine */

const MACHINE_TASKS = [
  { t: "cheetah-run", why: "drifts slowly" },
  { t: "pygame-pong", why: "breaks almost at once" },
  { t: "pygame-point-maze-var1", why: "never breaks" },
];

export function DreamMachine() {
  const [task, setTask] = useState(MACHINE_TASKS[0].t);
  const d = dreams[task];
  const { t, setT, playing, setPlaying } = useTicker(d.T, 260, 8);
  const COLS = 7;
  const first = Math.max(1, Math.min(t - COLS + 1, d.T - COLS + 1));
  const cols = Array.from({ length: COLS }, (_, k) => first + k).filter((s) => s <= t);
  const broken = d.onset != null && t >= d.onset;

  let story: string;
  if (d.onset == null) story = t < 20 ? "Each imagined frame is built on the one before it." : "This dream stays close to reality all the way: the gap never crosses the red line.";
  else if (t < Math.max(2, d.onset - 6)) story = "Each imagined frame is built on the one before it. So far the two rows look alike, and the gap bars stay short.";
  else if (t < d.onset) story = "Small errors pile up, like a photocopy of a photocopy. The gap bars grow toward the red line.";
  else story = `Broken since step ${d.onset}: the imagined frames no longer match what really happened, and every later frame is built on the mistake. The gap can dip back under the line for a moment, but the dream does not recover.`;

  return (
    <figure className="fig explainer">
      <div className="explainer-head">
        <div className="seg" role="group" aria-label="Which dream">
          {MACHINE_TASKS.map((m) => (
            <button key={m.t} aria-pressed={task === m.t} onClick={() => setTask(m.t)}>{pretty(m.t)} <span style={{ opacity: 0.7 }}>({m.why})</span></button>
          ))}
        </div>
        <button onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play"}</button>
        <button onClick={() => { setT(1); setPlaying(true); }}>Start over</button>
      </div>

      <div className="machine">
        <div className="m-inputs">
          <div className="stack" aria-hidden="true">{Array.from({ length: 8 }, (_, k) => <span key={k} style={{ transform: `translate(${k * 4}px, ${k * 4}px)` }} />)}</div>
          <p><b>8 real frames</b><br />and the actions taken</p>
        </div>
        <div className="m-arrow" aria-hidden="true">→</div>
        <div className={`m-engine${playing ? " pulse" : ""}`} key={t}>
          <b>World model</b>
          <span>imagines the next frame</span>
        </div>
        <div className="m-arrow" aria-hidden="true">→</div>

        <div className="m-strips">
          <div className="m-row m-steps"><span />{cols.map((s) => <span key={s}>step {s}</span>)}</div>
          <div className="m-row"><span className="m-lab r">What really happened</span>{cols.map((s) => <div key={s} className="m-cell"><Thumb task={task} kind="real" step={s} /></div>)}</div>
          <div className="m-row"><span className="m-lab d">What it imagined</span>{cols.map((s) => (
            <div key={s} className={`m-cell${s === t ? " newest" : ""}${d.onset != null && s >= d.onset ? " broken" : ""}`}><Thumb task={task} kind="dream" step={s} /></div>
          ))}</div>
          <div className="m-row m-gaps"><span className="m-lab">Gap between them</span>{cols.map((s) => {
            const g = d.err[s - 1], h = Math.min(g / 1.2, 1);
            const bad = g >= 0.5;
            return (
              <div key={s} className="gapbar" title={`gap ${g.toFixed(2)}x`}>
                <i style={{ height: `${h * 100}%`, background: bad ? "var(--broken)" : "var(--ink-2)" }} />
                <em style={{ bottom: `${(0.5 / 1.2) * 100}%` }} />
              </div>
            );
          })}</div>
          <svg className="m-loop" viewBox="0 0 100 10" preserveAspectRatio="none" aria-hidden="true">
            <path d="M99 1 V6 H1" fill="none" stroke="var(--dream)" strokeWidth="0.6" vectorEffect="non-scaling-stroke" strokeDasharray="4 3" />
          </svg>
          <p className="m-loop-lab">↩ each imagined frame is fed back in to imagine the next one</p>
        </div>
      </div>

      <p className={`m-story${broken ? " bad" : ""}`} aria-live="polite">{story}</p>
      <figcaption className="note">
        Real frames from the paper's world model. The model saw 8 real frames before step 1 (not shown) and was given the real actions throughout; it never sees the top row while imagining. The red tick on each gap bar is the break line, explained next; a bar turns orange when the gap is above it.
      </figcaption>
    </figure>
  );
}

/* ======================================================================== 2. when is a dream broken? */

// An illustration, not data: a friend walks a loop through a city; a guess of where they are drifts away.
const N = 64;
const RULER = 150;                                    // the typical gap between two random spots, in SVG units
function friendAt(i: number) {
  const a = (i / N) * Math.PI * 2;
  return { x: 200 + 120 * Math.cos(a) + 25 * Math.cos(3 * a), y: 135 + 75 * Math.sin(a) };
}
const GAPS = Array.from({ length: N }, (_, i) => {
  const base = 0.08 + 0.55 * Math.pow(i / N, 1.7);
  return Math.max(0.03, base + 0.05 * Math.sin(i * 0.9) + 0.03 * Math.sin(i * 2.3));
});
const ONSET = (() => {
  for (let i = 0; i + 4 < N; i++) if ([0, 1, 2, 3, 4].every((k) => GAPS[i + k] >= 0.5)) return i + 1;
  return null;
})();
function guessAt(i: number) {
  const f = friendAt(i), ang = 0.6 + i * 0.07;
  const r = GAPS[i] * RULER;
  return { x: f.x + r * Math.cos(ang), y: f.y + r * Math.sin(ang) };
}

export function BrokenExplainer() {
  const { t, setT, playing, setPlaying } = useTicker(N, 230, 10);
  const i = t - 1;
  const f = friendAt(i), g = guessAt(i);
  const broken = ONSET != null && t >= ONSET;
  const trail = Array.from({ length: t }, (_, k) => friendAt(k));

  return (
    <figure className="fig explainer">
      <div className="explainer-head">
        <p className="chart-title" style={{ margin: 0 }}>An everyday version: guessing where a friend is</p>
        <button onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play"}</button>
        <button onClick={() => { setT(1); setPlaying(true); }}>Start over</button>
      </div>
      <div className="two-up" style={{ margin: 0, alignItems: "start" }}>
        <div className="chart">
          <svg viewBox="0 0 400 290" role="img" aria-label="A friend walking a loop through a city and a guess of their position drifting away">
            <rect x="4" y="4" width="392" height="282" rx="10" fill="var(--card)" stroke="var(--rule)" />
            {Array.from({ length: 9 }, (_, k) => <line key={"v" + k} x1={30 + k * 43} x2={30 + k * 43} y1={10} y2={280} stroke="var(--grid)" strokeWidth="5" />)}
            {Array.from({ length: 6 }, (_, k) => <line key={"h" + k} y1={25 + k * 48} y2={25 + k * 48} x1={10} x2={390} stroke="var(--grid)" strokeWidth="5" />)}
            {/* the ruler: distance between two random spots */}
            <g>
              <circle cx={30} cy={262} r={4} fill="var(--ink-2)" />
              <circle cx={30 + RULER} cy={262} r={4} fill="var(--ink-2)" />
              <line x1={30} x2={30 + RULER} y1={262} y2={262} stroke="var(--ink-2)" strokeWidth="1.5" />
              <text className="label-2" x={30 + RULER + 8} y={266}>ruler: two random spots (1x)</text>
            </g>
            <path d={trail.map((p, k) => `${k ? "L" : "M"}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join("")} fill="none" stroke="var(--real)" strokeOpacity="0.35" strokeWidth="2" />
            {/* half a ruler around the friend: inside = close enough */}
            <circle cx={f.x} cy={f.y} r={RULER / 2} fill={broken ? "var(--broken-wash)" : "none"} stroke="var(--broken)" strokeDasharray="4 4" strokeWidth="1.25" />
            <text className="label-2" x={f.x} y={f.y - RULER / 2 - 6} textAnchor="middle">half a ruler</text>
            <line x1={f.x} y1={f.y} x2={g.x} y2={g.y} stroke="var(--ink)" strokeDasharray="2 3" />
            <circle cx={f.x} cy={f.y} r={7} fill="var(--real)" stroke="var(--paper)" strokeWidth="2" />
            <circle cx={g.x} cy={g.y} r={7} fill="var(--dream)" stroke="var(--paper)" strokeWidth="2" />
            <text className="label" x={f.x + 10} y={f.y - 10}>friend</text>
            <text className="label" x={g.x + 10} y={g.y + 16}>your guess</text>
          </svg>
        </div>
        <div>
          <LineChart ariaLabel="How far the guess is from the friend, drawn as the animation plays"
            series={[{ key: "g", label: "gap", values: GAPS.map((v, k) => (k < t ? v : null)), color: broken ? "var(--broken)" : "var(--ink)", width: 2.25 }]}
            xDomain={[1, N]} yDomain={[0, 1]} xTicks={[1, 16, 32, 48, 64]} yTicks={[0, 0.5, 1]} yFormat={(v) => `${v}x`}
            refs={[{ y: 0.5, label: "half a ruler: the break line", color: "var(--broken)", anchor: "start" },
              ...(broken && ONSET ? [{ x: ONSET, label: "lost", color: "var(--broken)", dash: "2 3" }] : [])]}
            shadeFrom={broken ? ONSET : null} cursor={t} height={250} xTitle="minutes" tipX={(x) => `minute ${x}`} tipValue={(v) => `${v.toFixed(2)}x`} />
          <p className={`m-story${broken ? " bad" : ""}`} style={{ marginTop: "0.5rem" }}>
            {broken
              ? "Your guess has been more than half a ruler away for 5 minutes in a row: you have lost your friend."
              : t < N / 3 ? "Your guess is close: well inside the dashed circle." : "Your guess drifts. The gap climbs toward half a ruler."}
          </p>
        </div>
      </div>
      <figcaption className="note">
        An illustration, not data. Why a ruler? "50 metres off" is close in a city and far in a room. Measuring the gap against the distance between two random spots in the same place makes "lost" mean the same thing everywhere.
      </figcaption>
    </figure>
  );
}

/* ---------------------------------------------- three real dreams on one scale */

const THREE = [
  { t: "cheetah-run", what: "drifts slowly, breaks at step 37" },
  { t: "pygame-pong", what: "breaks at step 4, almost at once" },
  { t: "pygame-point-maze-var1", what: "never crosses the line" },
];

export function ThreeDreams() {
  const { t } = useTicker(96, 120, 20);
  return (
    <figure className="fig">
      <p className="chart-title">Three real dreams, on one scale</p>
      <p className="chart-sub">The same measurement in a real dream: the gap between imagined and real, in rulers. Here the ruler is the gap between two random moments of the same task.</p>
      <div className="three">
        {THREE.map(({ t: task, what }) => {
          const d = dreams[task];
          const shown = d.err.map((v, k) => (k < t ? v : null));
          const s = Math.min(t, d.T);
          return (
            <div key={task}>
              <div className="pair-sm">
                <div><span className="r">real</span><Thumb task={task} kind="real" step={s} /></div>
                <div className={d.onset != null && s >= d.onset ? "broken" : ""}><span className="d">imagined</span><Thumb task={task} kind="dream" step={s} /></div>
              </div>
              <LineChart ariaLabel={`Gap between imagined and real frames for ${pretty(task)}`}
                series={[{ key: "e", label: "gap", values: shown, color: "var(--ink)", width: 2 }]}
                xDomain={[1, 96]} yDomain={[0, 1.2]} xTicks={[1, 48, 96]} yTicks={[0, 0.5, 1]} yFormat={(v) => `${v}x`}
                refs={[{ y: 0.5, color: "var(--broken)" }]} shadeFrom={d.onset != null && t >= d.onset ? d.onset : null}
                cursor={t} height={150} tipX={(x) => `step ${x}`} tipValue={(v) => `${v.toFixed(2)}x`} />
              <p className="three-lab"><b>{pretty(task)}</b><br />{what}</p>
            </div>
          );
        })}
      </div>
      <figcaption className="note">A dream is broken once its gap stays above the red line for 5 steps in a row. A single spike above the line does not count; the gap is smoothed over 5 steps first.</figcaption>
    </figure>
  );
}
