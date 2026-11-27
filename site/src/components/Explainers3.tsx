import { useEffect, useMemo, useRef, useState } from "react";
import { dreams, pretty, seenTasks, unseenTasks } from "../data";
import { Thumb } from "./Figures";
import LineChart from "./LineChart";

const reduceMotion = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
const pct = (v: number) => `${Math.round(v * 100)}%`;

function useTick(n: number, ms: number, hold = 8) {
  const [t, setT] = useState(reduceMotion ? n : 1);
  const [playing, setPlaying] = useState(!reduceMotion);
  const h = useRef(0);
  useEffect(() => { setT(reduceMotion ? n : 1); }, [n]);
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => setT((x) => (x < n ? x + 1 : h.current++ < hold ? x : ((h.current = 0), 1))), ms);
    return () => window.clearInterval(id);
  }, [playing, n, ms, hold]);
  return { t, setT, playing, setPlaying };
}

/* ======================================================================== how a prediction is made */

function SCurve({ p, color, label, dashed }: { p: number; color: string; label: string; dashed?: boolean }) {
  const W = 260, H = 150, l = 34, r = 10, top = 10, b = 26;
  const iw = W - l - r, ih = H - top - b;
  const z0 = -6, z1 = 6;
  const X = (z: number) => l + ((z - z0) / (z1 - z0)) * iw, Y = (q: number) => top + ih - q * ih;
  const sig = (z: number) => 1 / (1 + Math.exp(-z));
  const curve = Array.from({ length: 61 }, (_, k) => { const z = z0 + (k / 60) * (z1 - z0); return `${k ? "L" : "M"}${X(z).toFixed(1)} ${Y(sig(z)).toFixed(1)}`; }).join("");
  const z = Math.log(Math.max(p, 1e-4) / Math.max(1 - p, 1e-4));
  const zc = Math.max(z0, Math.min(z1, z));
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`${label}: ${pct(p)}`}>
        <g className="grid">{[0, 0.5, 1].map((q) => <line key={q} x1={l} x2={l + iw} y1={Y(q)} y2={Y(q)} />)}</g>
        <g className="axis">{[0, 0.5, 1].map((q) => <text key={q} x={l - 5} y={Y(q) + 4} textAnchor="end">{pct(q)}</text>)}</g>
        <path d={curve} fill="none" stroke={color} strokeWidth="2.5" strokeDasharray={dashed ? "7 5" : undefined} />
        <line x1={X(zc)} x2={X(zc)} y1={top + ih} y2={Y(sig(zc))} stroke="var(--ink-2)" strokeDasharray="2 3" />
        <line x1={l} x2={X(zc)} y1={Y(sig(zc))} y2={Y(sig(zc))} stroke="var(--ink-2)" strokeDasharray="2 3" />
        <circle cx={X(zc)} cy={Y(sig(zc))} r="7" fill={color} stroke="var(--card)" strokeWidth="2.5" style={{ transition: "cx 200ms, cy 200ms" }} />
        <text className="axis-title" x={l + iw / 2} y={H - 4} textAnchor="middle">score: the weighed sum of the 11 numbers</text>
      </svg>
    </div>
  );
}

export function PredictionExplainer() {
  const task = "cheetah-run";
  const d = dreams[task];
  const { t, playing, setPlaying, setT } = useTick(d.T, 280, 10);
  const i = t - 1;
  const broken = d.onset != null && t >= d.onset;
  const bars = [
    { k: "round trip", v: d.u_r[i], max: 0.7, c: "var(--s-ur)" },
    { k: "second thoughts", v: d.u_f[i], max: 0.25, c: "var(--s-uf)" },
    { k: "two tries disagree", v: d.u_s[i], max: 0.05, c: "var(--s-us)" },
    { k: "step", v: t, max: 96, c: "var(--ink-2)" },
  ];
  return (
    <figure className="fig explainer">
      <div className="explainer-head">
        <p className="chart-title" style={{ margin: 0 }}>Cheetah run, step {t}{broken ? `: broken since step ${d.onset}` : ""}</p>
        <button onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play"}</button>
        <button onClick={() => { setT(1); setPlaying(true); }}>Start over</button>
      </div>
      <div className="pred">
        <div className="pred-frames">
          <div><span className="r">real</span><Thumb task={task} kind="real" step={t} /></div>
          <div className={broken ? "broken" : ""}><span className="d">imagined</span><Thumb task={task} kind="dream" step={t} /></div>
          <p className="muted">The predictions never see the real frame. They see only the numbers.</p>
        </div>
        <div className="pred-arrow" aria-hidden="true">→</div>
        <div className="pred-bars">
          <p className="pred-h">The numbers at this step</p>
          {bars.map((b) => (
            <div key={b.k} className="pbar"><span>{b.k}</span><div><i style={{ width: `${Math.min(1, b.v / b.max) * 100}%`, background: b.c }} /></div></div>
          ))}
          <p className="muted" style={{ fontSize: 12, margin: "0.4rem 0 0" }}>Plus each signal's recent average and trend, and scene motion: 11 numbers in all.</p>
        </div>
        <div className="pred-arrow" aria-hidden="true">→</div>
        <div className="pred-curves">
          <div>
            <p className="pred-h"><span className="key"><i className="solid" /></span> Forecast: breaks within the next 8 steps</p>
            {broken ? <p className="muted pred-done">The dream has broken, so this question no longer applies.</p> : <><SCurve p={d.p_forecast[i]} color="var(--dream)" label="Forecast" /><p className="pred-v" style={{ color: "var(--dream)" }}>{pct(d.p_forecast[i])}</p></>}
          </div>
          <div>
            <p className="pred-h"><span className="key"><i className="dashed" /></span> Alarm: has already broken</p>
            <SCurve p={d.p_detect[i]} color="var(--real)" label="Alarm" dashed />
            <p className="pred-v" style={{ color: "var(--real)" }}>{pct(d.p_detect[i])}</p>
          </div>
        </div>
      </div>
      <figcaption className="note">
        Each prediction weighs the 11 numbers, adds them into one score, and an S-shaped curve turns the score into a chance between 0% and 100%. This is logistic regression. The weights were learned from 320 other dreams (the training set) and never from the dreams shown here.
      </figcaption>
    </figure>
  );
}

/* ======================================================================== the pairs game */

type Moment = { task: string; step: number; p: number; y: number };

function moments(kind: "alarm" | "forecast"): Moment[] {
  const out: Moment[] = [];
  for (const task of seenTasks) {
    const d = dreams[task];
    for (let s = 1; s <= d.T; s++) {
      const brokenNow = d.onset != null && s >= d.onset;
      if (kind === "alarm") out.push({ task, step: s, p: d.p_detect[s - 1], y: brokenNow ? 1 : 0 });
      else if (!brokenNow) out.push({ task, step: s, p: d.p_forecast[s - 1], y: d.onset != null && d.onset - s <= 8 ? 1 : 0 });
    }
  }
  return out;
}

function auroc(ms: Moment[]) {
  const pos = ms.filter((m) => m.y === 1).map((m) => m.p), neg = ms.filter((m) => m.y === 0).map((m) => m.p);
  let w = 0;
  for (const a of pos) for (const b of neg) w += a > b ? 1 : a === b ? 0.5 : 0;
  return w / (pos.length * neg.length);
}

export function PairsGame() {
  const [kind, setKind] = useState<"alarm" | "forecast">("alarm");
  const ms = useMemo(() => moments(kind), [kind]);
  const pos = useMemo(() => ms.filter((m) => m.y === 1), [ms]);
  const neg = useMemo(() => ms.filter((m) => m.y === 0), [ms]);
  const full = useMemo(() => auroc(ms), [ms]);
  const [round, setRound] = useState(0);
  const [wins, setWins] = useState(0);
  const [pair, setPair] = useState<[Moment, Moment] | null>(null);
  const seed = useRef(7);
  const rnd = () => ((seed.current = (seed.current * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);

  useEffect(() => { setRound(0); setWins(0); setPair(null); seed.current = 7; }, [kind]);
  useEffect(() => {
    if (reduceMotion) return;
    const id = window.setInterval(() => {
      setRound((r) => {
        if (r >= 60) { setWins(0); setPair(null); return 0; }
        const a = pos[Math.floor(rnd() * pos.length)], b = neg[Math.floor(rnd() * neg.length)];
        setPair([a, b]);
        setWins((w) => w + (a.p > b.p ? 1 : a.p === b.p ? 0.5 : 0));
        return r + 1;
      });
    }, 900);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pos, neg]);

  const name = kind === "alarm" ? "Alarm" : "Forecast";
  const posLab = kind === "alarm" ? "broken" : "breaks within 8";
  const negLab = kind === "alarm" ? "not broken" : "does not";
  const won = pair ? pair[0].p > pair[1].p : false;

  return (
    <figure className="fig explainer">
      <div className="explainer-head">
        <p className="chart-title" style={{ margin: 0 }}>The scoring game, for the</p>
        <div className="seg" role="group" aria-label="Which prediction">
          <button aria-pressed={kind === "alarm"} onClick={() => setKind("alarm")}>Alarm</button>
          <button aria-pressed={kind === "forecast"} onClick={() => setKind("forecast")}>Forecast</button>
        </div>
      </div>
      <div className="game">
        <div className="game-pair">
          {pair ? (
            <>
              {[pair[0], pair[1]].map((m, k) => (
                <div key={k} className={`gm${k === 0 ? " pos" : ""}`}>
                  <span className="gm-lab">{k === 0 ? posLab : negLab}</span>
                  <div className="mini2"><Thumb task={m.task} kind="real" step={m.step} /><Thumb task={m.task} kind="dream" step={m.step} /></div>
                  <span className="muted">{pretty(m.task)}, step {m.step}</span>
                  <b style={{ color: kind === "alarm" ? "var(--real)" : "var(--dream)" }}>{name}: {pct(m.p)}</b>
                </div>
              ))}
              <div className={`verdict ${won ? "win" : "lose"}`}>{won ? "✓ higher for the " + posLab + " one" : "✗ the other one got the higher chance"}</div>
            </>
          ) : <p className="muted">Drawing the first pair...</p>}
        </div>
        <div className="game-tally">
          <p className="tally-n">{round ? pct(wins / round) : "..."}</p>
          <p className="muted">of {round} pairs won so far</p>
          <div className="meter-bar"><i style={{ width: `${round ? (wins / round) * 100 : 0}%`, background: kind === "alarm" ? "var(--real)" : "var(--dream)" }} /></div>
          <p className="muted" style={{ fontSize: 12 }}>Over every possible pair of these 20 example dreams: <b style={{ color: "var(--ink)" }}>{pct(full)}</b>. 50% would be a coin flip.</p>
        </div>
      </div>
      <figcaption className="note">
        Draw one moment where the answer is yes and one where it is no. If the {name} gives the yes moment the higher chance, it wins the pair. The share of pairs won is called the AUROC. This game uses only the 20 example dreams on this page; the scores in the next chart use all 160 test dreams.
      </figcaption>
    </figure>
  );
}

/* ======================================================================== lining dreams up at their break */

const ALIGN = ["atari-ms-pacman", "bipedal-walker-uneven", "atari-pong", "cheetah-run", "walker-walk"];

export function AlignExplainer() {
  const { t } = useTick(60, 90, 25);
  // 0..20: own time; 20..40: slide; 40..60: hold aligned
  const f = Math.max(0, Math.min(1, (t - 20) / 20));
  const ease = f < 0.5 ? 2 * f * f : 1 - Math.pow(-2 * f + 2, 2) / 2;
  const W = 720, H = 240, l = 40, r = 20, top = 12, b = 34;
  const iw = W - l - r, ih = H - top - b;
  const xMin = -70, xMax = 96;
  const X = (x: number) => l + ((x - xMin) / (xMax - xMin)) * iw;
  const Y = (v: number) => top + ih - Math.min(v, 1.2) / 1.2 * ih;
  const colors = ["#7fb6ff", "#e8a0ff", "#ffd27a", "#9be7c4", "#ff9f80"];
  return (
    <figure className="fig explainer">
      <p className="chart-title">{ease < 0.05 ? "Five real dreams, each breaking at its own time" : ease > 0.95 ? "Lined up so every break sits at 0" : "Sliding each dream so its break lines up..."}</p>
      <div className="chart">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Gap curves of five dreams sliding so that their breaks line up">
          <g className="grid">{[0, 0.5, 1].map((v) => <line key={v} x1={l} x2={l + iw} y1={Y(v)} y2={Y(v)} />)}</g>
          <g className="axis">{[0, 0.5, 1].map((v) => <text key={v} x={l - 6} y={Y(v) + 4} textAnchor="end">{v}x</text>)}</g>
          <line x1={l} x2={l + iw} y1={Y(0.5)} y2={Y(0.5)} stroke="var(--broken)" strokeDasharray="4 4" />
          {ease > 0.6 && <><line x1={X(0)} x2={X(0)} y1={top} y2={top + ih} stroke="var(--ink)" strokeDasharray="3 3" /><text className="label" x={X(0) + 5} y={top + 12}>the break</text></>}
          {ALIGN.map((task, k) => {
            const d = dreams[task], shift = d.onset! * ease;
            const path = d.err.map((v, j) => `${j ? "L" : "M"}${X(j + 1 - shift).toFixed(1)} ${Y(v).toFixed(1)}`).join("");
            const ox = X(d.onset! - shift);
            return (
              <g key={task}>
                <path d={path} fill="none" stroke={colors[k]} strokeWidth="1.6" opacity="0.9" />
                <circle cx={ox} cy={Y(d.err[d.onset! - 1])} r="4.5" fill={colors[k]} stroke="var(--card)" strokeWidth="2" />
              </g>
            );
          })}
          <g className="axis">{(ease > 0.6 ? [-60, -40, -20, 0, 20, 40, 60, 80] : [1, 20, 40, 60, 80, 96]).map((v) => <text key={v} x={X(v)} y={top + ih + 16} textAnchor="middle">{ease > 0.6 && v > 0 ? `+${v}` : v}</text>)}</g>
          <text className="axis-title" x={l + iw / 2} y={H - 2} textAnchor="middle">{ease > 0.6 ? "steps from the break" : "dream step"}</text>
        </svg>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.3rem 1.1rem", fontSize: "var(--t-small)", marginTop: "0.4rem" }}>
        {ALIGN.map((task, k) => <span key={task} className="key"><i style={{ background: colors[k], height: 3 }} />{pretty(task)}, breaks at step {dreams[task].onset}</span>)}
      </div>
      <figcaption className="note">Dots mark each dream's break. Once they are lined up, we can ask what the signals were doing 1, 2, 5 steps before a break, compared with dreams of the same task that had not broken at that step.</figcaption>
    </figure>
  );
}

/* ======================================================================== survival lanes */

export function SurvivalLanes() {
  const [which, setWhich] = useState<"seen" | "unseen">("seen");
  const tasks = which === "seen" ? seenTasks : unseenTasks;
  const { t } = useTick(96, 110, 18);
  const alive = (task: string, s: number) => { const d = dreams[task]; return d.onset == null || s < d.onset; };
  const watched = (task: string, s: number) => s <= dreams[task].T;
  // share still sound among dreams still watched, chained step by step (Kaplan-Meier)
  const S: number[] = [];
  let sv = 1;
  for (let s = 1; s <= 96; s++) {
    const atRisk = tasks.filter((x) => watched(x, s) && (dreams[x].onset == null || dreams[x].onset! >= s)).length;
    const breaks = tasks.filter((x) => dreams[x].onset === s).length;
    if (atRisk > 0) sv *= 1 - breaks / atRisk;
    S.push(sv);
  }
  const nowSound = tasks.filter((x) => alive(x, t) && watched(x, t)).length;
  const nowWatched = tasks.filter((x) => watched(x, t)).length;
  return (
    <figure className="fig explainer">
      <div className="explainer-head">
        <p className="chart-title" style={{ margin: 0 }}>One lane per example dream</p>
        <div className="seg" role="group" aria-label="Which tasks">
          <button aria-pressed={which === "seen"} onClick={() => setWhich("seen")}>20 familiar tasks</button>
          <button aria-pressed={which === "unseen"} onClick={() => setWhich("unseen")}>10 unseen tasks</button>
        </div>
        <span className="muted">step {t}: {nowSound} of {nowWatched} still sound</span>
      </div>
      <div className="two-up" style={{ margin: 0, alignItems: "start" }}>
        <div className="lanes">
          {tasks.map((task) => {
            const d = dreams[task];
            const end = Math.min(t, d.T);
            const on = d.onset;
            const goodTo = on == null ? end : Math.min(end, on - 1);
            return (
              <div key={task} className="lane">
                <span>{pretty(task)}</span>
                <div className="lane-bar">
                  <i className="ok" style={{ width: `${(goodTo / 96) * 100}%` }} />
                  {on != null && end >= on && <i className="bad" style={{ left: `${((on - 1) / 96) * 100}%`, width: `${((end - on + 1) / 96) * 100}%` }} />}
                  {d.T < 96 && t > d.T && <em style={{ left: `${(d.T / 96) * 100}%` }} title="watched only this long" />}
                  <b style={{ left: `${(t / 96) * 100}%` }} />
                </div>
              </div>
            );
          })}
        </div>
        <div>
          <LineChart ariaLabel="Share of example dreams still sound, drawn as the lanes play"
            series={[{ key: "s", label: "still sound", values: S.map((v, k) => (k < t ? v : null)), color: which === "seen" ? "var(--real)" : "var(--dream)", width: 2.5, step: true }]}
            xDomain={[1, 96]} yDomain={[0, 1]} xTicks={[1, 16, 32, 48, 64, 80, 96]} yTicks={[0, 0.5, 1]} yFormat={(v) => pct(v)}
            refs={[{ y: 0.5, label: "half broken" }]} cursor={t} height={260} xTitle="dream step" tipX={(x) => `step ${x}`} tipValue={pct} />
        </div>
      </div>
      <figcaption className="note">
        Teal: still sound. Orange: broken. A grey tick marks where a dream stopped being watched (pick cube and push banana episodes are only 17 steps long); such a dream counts while watched and then leaves the count, rather than being assumed to break or to last forever. Multiplying "survived this step, given it survived the last" is the Kaplan-Meier curve.
      </figcaption>
    </figure>
  );
}
