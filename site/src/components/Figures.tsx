import { useEffect, useRef, useState } from "react";
import LineChart from "./LineChart";
import { useWidth } from "./useWidth";
import { dreams, drawFrame, frameUrl, loadSprite, pretty, site, unseenTasks } from "../data";

const f2 = (v: number) => v.toFixed(2);

/* ------------------------------------------------------------------ AUROC three views */

type View = "auroc" | "beyond_time" | "within_task_and_time";
const VIEW_LABEL: Record<View, string> = {
  auroc: "Any two moments",
  beyond_time: "Same step",
  within_task_and_time: "Same task and step",
};

export function AurocViews() {
  const [view, setView] = useState<View>("auroc");
  const v = site.views;
  const rows = [
    { k: "Alarm: has it already broken?", who: "familiar", d: v.seen.detection, c: "var(--real)" },
    { k: "", who: "unseen", d: v.unseen.detection, c: "var(--real)" },
    { k: "Forecast: will it break within 8 steps?", who: "familiar", d: v.seen.forecast, c: "var(--dream)" },
    { k: "", who: "unseen", d: v.unseen.forecast, c: "var(--dream)" },
  ];
  const [box, W] = useWidth<HTMLDivElement>(640);
  const narrow = W < 480;
  const l = narrow ? 0 : 190, r = 40, rowH = narrow ? 62 : 46, top = 28, H = top + rows.length * rowH + 34;
  const X = (a: number) => l + ((a - 0.4) / 0.6) * (W - l - r);
  return (
    <figure className="fig">
      <p className="chart-title">How often each prediction picks the broken dream</p>
      <p className="chart-sub">Take one moment where a dream has broken and one where it has not. How often does the prediction give the broken one the higher chance? This share is called AUROC: 0.5 is a coin flip, 1.0 is perfect. The buttons make the two moments more and more alike. Scored on the 160 test dreams.</p>
      <div className="seg" role="group" aria-label="Which dreams are compared">
        {(Object.keys(VIEW_LABEL) as View[]).map((k) => (
          <button key={k} aria-pressed={view === k} onClick={() => setView(k)}>{VIEW_LABEL[k]}</button>
        ))}
      </div>
      <div className="chart" ref={box} style={{ marginTop: "0.75rem" }}>
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="AUROC for detection and forecast, familiar and unseen tasks">
          <g className="grid">{[0.5, 0.6, 0.7, 0.8, 0.9, 1].map((t) => <line key={t} x1={X(t)} x2={X(t)} y1={top - 8} y2={H - 26} />)}</g>
          <g className="axis">{[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1].map((t) => <text key={t} x={X(t)} y={H - 10} textAnchor="middle">{t.toFixed(1)}</text>)}</g>
          <line x1={X(0.5)} x2={X(0.5)} y1={top - 14} y2={H - 26} stroke="var(--ink-2)" strokeDasharray="4 4" />
          <text className="label-2" x={X(0.5) + 5} y={top - 6}>coin flip</text>
          {rows.map((row, i) => {
            const y = top + i * rowH + rowH / 2 + (narrow ? 10 : 0);
            const ty = narrow ? y - 22 : y;
            const a = row.d[view] as number, a0 = row.d.auroc as number;
            return (
              <g key={i}>
                {narrow
                  ? <text className="label-2" x={0} y={ty}>{row.k ? <tspan style={{ fontWeight: 600, fill: "var(--ink)" }}>{row.k.split(":")[0]}, </tspan> : null}{row.who} tasks</text>
                  : <>
                      {row.k && <text className="label" x={0} y={y - 4} style={{ fontWeight: 600 }}>{row.k.split(":")[0]}</text>}
                      <text className="label-2" x={0} y={y + (row.k ? 12 : 4)}>{row.who} tasks</text>
                    </>}
                <line x1={X(0.4)} x2={X(1)} y1={y} y2={y} stroke="var(--grid)" />
                {view !== "auroc" && <circle cx={X(a0)} cy={y} r={5} fill="none" stroke={row.c} strokeWidth={1.5} opacity={0.5} />}
                <g className="dots-row" style={{ transform: `translateX(${X(a)}px)` }}>
                  <circle cx={0} cy={y} r={7} fill={row.c} stroke="var(--paper)" strokeWidth={2} />
                  <text className="label" x={12} y={y + 4} style={{ fontWeight: 600 }}>{f2(a)}</text>
                </g>
              </g>
            );
          })}
        </svg>
      </div>
      <figcaption className="note">
        Hollow circles mark the "any two moments" result. That result mixes three skills: knowing which tasks are risky, knowing that late steps are riskier, and reading the dream in front of you. Only the last view isolates that third skill. The Alarm keeps about 0.72 on familiar tasks; the Forecast keeps 0.61, and on unseen tasks 0.56.
      </figcaption>
    </figure>
  );
}

/* ------------------------------------------------------------------ warning time */

export function WarningHist() {
  const w: number[] = site.warning_times;
  const bins = [
    { k: "0", f: (x: number) => x === 0 },
    { k: "1-4", f: (x: number) => x >= 1 && x <= 4 },
    { k: "5-9", f: (x: number) => x >= 5 && x <= 9 },
    { k: "10-19", f: (x: number) => x >= 10 && x <= 19 },
    { k: "20-39", f: (x: number) => x >= 20 && x <= 39 },
    { k: "40+", f: (x: number) => x >= 40 },
  ].map((b) => ({ ...b, n: w.filter(b.f).length }));
  const [box, W] = useWidth<HTMLDivElement>(640);
  const H = 230, l = 32, r = 4, t = 16, b = 40, iw = W - l - r, ih = H - t - b;
  const max = 70, bw = iw / bins.length;
  const [hov, setHov] = useState<number | null>(null);
  return (
    <figure className="fig">
      <p className="chart-title">Steps of warning before each break</p>
      <p className="chart-sub">For each of the {w.length} familiar test dreams that broke after step 1: how many steps before the break the forecast first passed 50%.</p>
      <div className="chart" ref={box} style={{ position: "relative" }}>
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="Histogram of warning times; most are zero">
          <g className="grid">{[0, 20, 40, 60].map((v) => <line key={v} x1={l} x2={W - r} y1={t + ih - (v / max) * ih} y2={t + ih - (v / max) * ih} />)}</g>
          <g className="axis">{[0, 20, 40, 60].map((v) => <text key={v} x={l - 6} y={t + ih - (v / max) * ih + 4} textAnchor="end">{v}</text>)}</g>
          {bins.map((bn, i) => {
            const h = (bn.n / max) * ih, x = l + i * bw + 6, y = t + ih - h;
            const fill = i === 0 ? "var(--broken)" : "var(--dream)";
            return (
              <g key={bn.k} onPointerEnter={() => setHov(i)} onPointerLeave={() => setHov(null)}>
                <rect x={l + i * bw} y={t} width={bw} height={ih} fill="transparent" />
                <path d={`M${x} ${t + ih}V${y + 4}q0 -4 4 -4H${x + bw - 16}q4 0 4 4V${t + ih}Z`.replace(/q0 -4 4 -4H/, h > 4 ? "q0 -4 4 -4H" : "H")} fill={fill} opacity={hov == null || hov === i ? 1 : 0.55} />
                <text className="label" x={x + (bw - 12) / 2} y={y - 6} textAnchor="middle">{bn.n}</text>
                <text className="axis-title" x={x + (bw - 12) / 2} y={t + ih + 16} textAnchor="middle">{bn.k}</text>
              </g>
            );
          })}
          <line className="baseline" x1={l} x2={W - r} y1={t + ih} y2={t + ih} />
          <text className="axis-title" x={l + iw / 2} y={H - 4} textAnchor="middle">steps of warning</text>
        </svg>
      </div>
      <figcaption className="note">
        {bins[0].n} of {w.length} breaks came with no warning at all; the median is 0 steps. A check we added for this page: 23 of the 24 warnings of 5 steps or more began at step 1 or 2 of the dream. Those are not warnings about a moment; the model was flagging a risky task from the start.
      </figcaption>
    </figure>
  );
}

/* ------------------------------------------------------------------ reliability */

export function Reliability() {
  const bins = site.reliability.bins as { lo: number; hi: number; n: number; mean_pred: number; frac_pos: number }[];
  const [box, bw] = useWidth<HTMLDivElement>(470);
  const W = Math.min(bw, 470), H = Math.min(400, W * 0.85), l = 44, r = 12, t = 26, b = 40, iw = W - l - r, ih = H - t - b;
  const X = (v: number) => l + v * iw, Y = (v: number) => t + ih - v * ih;
  const [hov, setHov] = useState<number | null>(null);
  return (
    <figure className="fig" style={{ margin: 0 }}>
      <p className="chart-title">Does "40%" mean 40%?</p>
      <p className="chart-sub">Forecasts grouped in bins of 0.1; dot area shows how many steps fall in the bin.</p>
      <div className="chart" ref={box} style={{ position: "relative" }}>
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="Reliability diagram of the forecast">
          <g className="grid">{[0, 0.5, 1].map((v) => <line key={v} x1={l} x2={l + iw} y1={Y(v)} y2={Y(v)} />)}</g>
          <g className="axis">
            {[0, 0.5, 1].map((v) => <text key={"y" + v} x={l - 6} y={Y(v) + 4} textAnchor="end">{v * 100}%</text>)}
            {[0, 0.5, 1].map((v) => <text key={"x" + v} x={X(v)} y={t + ih + 16} textAnchor="middle">{v * 100}%</text>)}
            <text className="axis-title" x={l + iw / 2} y={H - 4} textAnchor="middle">forecast probability</text>
          </g>
          <line x1={X(0)} y1={Y(0)} x2={X(1)} y2={Y(1)} stroke="var(--ink-2)" strokeDasharray="4 4" />
          <text className="label-2" x={X(0.62)} y={Y(0.72)} transform={`rotate(-${Math.atan(ih / iw) * 57.3} ${X(0.62)} ${Y(0.72)})`}>perfectly calibrated</text>
          <line className="baseline" x1={l} x2={l + iw} y1={t + ih} y2={t + ih} />
          {bins.map((bn, i) => (
            <circle key={i} cx={X(bn.mean_pred)} cy={Y(bn.frac_pos)} r={Math.max(4, Math.sqrt(bn.n) / 3.2)}
              fill="var(--dream)" fillOpacity={bn.n >= 50 ? 0.85 : 0.3} stroke="var(--paper)" strokeWidth={2}
              onPointerEnter={() => setHov(i)} onPointerLeave={() => setHov(null)} />
          ))}
          <text className="axis-title" x={l} y={t - 12}>share that broke</text>
        </svg>
        {hov != null && (
          <div className="tooltip" style={{ left: `${(X(bins[hov].mean_pred) / W) * 100}%`, top: `${(Y(bins[hov].frac_pos) / H) * 100}%` }}>
            said {Math.round(bins[hov].mean_pred * 100)}% · {Math.round(bins[hov].frac_pos * 100)}% broke · {bins[hov].n} steps
          </div>
        )}
      </div>
      <figcaption className="note">Average gap 0.07: each forecast's distance from how often its group really broke, averaged over all 6,453 forecasts. Below 30% the Forecast is close to calibrated. Between 30% and 60% it is overconfident: it said 35% to 55%, but only 18% to 35% broke (a largest gap of 0.27). Faded dots hold fewer than 50 steps.</figcaption>
    </figure>
  );
}

/* ------------------------------------------------------------------ T1 window sweep */

export function WindowSweep() {
  const ks = [1, 2, 4, 8, 16];
  const t1 = site.forecast_tests.T1_window_sweep;
  const xs = ks.map((k) => Math.log2(k));
  return (
    <figure className="fig" style={{ margin: 0 }}>
      <p className="chart-title">Is a short-range forecast easier?</p>
      <p className="chart-sub">The Forecast for "breaks within the next k steps", rated the same way (AUROC, 0.5 = coin flip), on familiar tasks.</p>
      <LineChart ariaLabel="Forecast AUROC by window length"
        width={470} height={360} rightPad={120}
        series={[
          { key: "bt", label: "same step", xs, values: ks.map((k) => t1[k].test.beyond_time), color: "var(--dream)", endLabel: true },
          { key: "wt", label: "same task and step", xs, values: ks.map((k) => t1[k].test.within_task_and_time), color: "var(--dream)", dash: "5 4", endLabel: true },
        ]}
        x0={0} xDomain={[0, 4]} yDomain={[0.4, 1]} xTicks={xs} yTicks={[0.5, 0.6, 0.7, 0.8, 0.9, 1]}
        xFormat={(v) => String(2 ** v)} tipX={(v) => `k = ${2 ** v}`} xTitle="k, steps ahead"
        refs={[{ y: 0.5, label: "coin flip" }]} />
      <figcaption className="note">If the signals gave a short warning, k = 1 would do best. It does worst within task and step (0.59 against 0.66 at k = 16).</figcaption>
    </figure>
  );
}

/* ------------------------------------------------------------------ T2 aligned at the break */

export function AlignedAtBreak() {
  const [mode, setMode] = useState<"test" | "explore">("test");
  const t2 = site.forecast_tests.T2_aligned.test;
  const ex = site.t2_latent_error_fit;
  const offs: number[] = t2.offsets;
  const sig = [
    { k: "u_r", c: "var(--s-ur)", l: "u_r" },
    { k: "u_f", c: "var(--s-uf)", l: "u_f" },
    { k: "u_s", c: "var(--s-us)", l: "u_s" },
  ];
  const series = mode === "test"
    ? sig.map((s, i) => ({
        key: s.k, label: s.l, xs: offs, values: t2.signals[s.k].mean, color: s.c, width: 1.75,
        band: i === 0 ? { lo: t2.signals[s.k].ci_lo, hi: t2.signals[s.k].ci_hi } : undefined, endLabel: true,
      }))
    : [
        ...sig.map((s) => ({ key: s.k, label: s.l, xs: ex.offsets, values: ex[s.k], color: s.c, width: 1.5, opacity: 0.75 })),
        { key: "err", label: "latent error", xs: ex.offsets, values: ex.latent_error, color: "var(--ink)", width: 2.75, endLabel: true },
      ];
  return (
    <figure className="fig">
      <p className="chart-title">Lined up at the break</p>
      <p className="chart-sub">
        How far each signal sits above unbroken dreams of the same task at the same step, in standard deviations, from 10 steps before the official break to 4 after.
      </p>
      <div className="seg" role="group" aria-label="Which analysis">
        <button aria-pressed={mode === "test"} onClick={() => setMode("test")}>Planned test: signals ({t2.n_dreams} test dreams)</button>
        <button aria-pressed={mode === "explore"} onClick={() => setMode("explore")}>Extra check: add the error itself ({ex.n_dreams} fit dreams)</button>
      </div>
      <div style={{ marginTop: "0.75rem" }}>
        <LineChart ariaLabel="Signals aligned at the break onset, compared with matched unbroken dreams"
          series={series} x0={-10} xDomain={[-10, 4]} yDomain={[-0.4, 1.2]} rightPad={90}
          xTicks={[-10, -8, -6, -4, -2, 0, 2, 4]} yTicks={[-0.4, 0, 0.4, 0.8, 1.2]}
          xFormat={(v) => (v > 0 ? `+${v}` : String(v))} xTitle="steps from the official break" yFormat={(v) => v.toFixed(1)}
          tipX={(v) => `${v > 0 ? "+" : ""}${v} steps`} tipValue={(v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)} SD`}
          refs={[{ x: 0, label: "official break" }, { y: 0, dash: "1 0", color: "var(--ink-2)" }]} height={280} />
      </div>
      <figcaption className="note">
        {mode === "test"
          ? "The rule written before the test: claim a lead only if the signals are clearly above normal (the 95% range of likely values stays above 0) at 2 or more steps before the break. They are: 2 steps before, round trip sits 0.58 standard deviations above normal (likely range +0.23 to +0.94, the shaded band). On unseen tasks there is no lead."
          : "Not in the plan, so exploratory. The error is already 0.5 SD above the controls 2 steps before the official onset and climbs at the same pace as the signals; after the onset the signals level off while the error keeps rising. The signals track error that is growing below the threshold."}
      </figcaption>
    </figure>
  );
}

/* ------------------------------------------------------------------ survival */

export function SurvivalCurves() {
  const sv = site.survival;
  const xs = Array.from({ length: 96 }, (_, i) => i + 1);
  const [hl, setHl] = useState<string | null>(null);
  const faint = (split: "seen" | "unseen", color: string) =>
    Object.entries(sv[split].tasks as Record<string, { S: number[] }>).map(([t, v]) => ({
      key: t, label: pretty(t), values: [1, ...v.S], xs: [0, ...xs.slice(0, v.S.length)], color,
      width: hl === t ? 2.25 : 1, opacity: hl == null ? 0.22 : hl === t ? 1 : 0.1, step: true, quiet: true,
    }));
  const series = [
    ...faint("seen", "var(--real)"), ...faint("unseen", "var(--dream)"),
    { key: "seen", label: "familiar", values: [1, ...sv.seen.pooled.S], xs: [0, ...xs], color: "var(--real)", width: 3, step: true, endLabel: true },
    { key: "unseen", label: "unseen", values: [1, ...sv.unseen.pooled.S], xs: [0, ...xs], color: "var(--dream)", width: 3, step: true, endLabel: true },
  ];
  return (
    <figure className="fig">
      <p className="chart-title">The trust horizon: share of dreams not yet broken</p>
      <p className="chart-sub">Kaplan-Meier curves, primary label. Thick lines pool all tasks; thin lines are single tasks. Hover a name below to find its curve.</p>
      <LineChart ariaLabel="Kaplan-Meier survival of dreams, familiar against unseen tasks"
        series={series} x0={0} xDomain={[0, 96]} yDomain={[0, 1]} rightPad={70}
        xTicks={[0, 16, 32, 48, 64, 80, 96]} yTicks={[0, 0.25, 0.5, 0.75, 1]} yFormat={(v) => `${v * 100}%`}
        xTitle="dream step" tipX={(v) => `step ${v}`} tipValue={(v) => `${Math.round(v * 100)}% unbroken`}
        refs={[{ y: 0.5, label: "half the dreams broken" }]} height={300} />
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.25rem 0.9rem", fontFamily: "var(--sans)", fontSize: "var(--t-small)", marginTop: "0.6rem" }}>
        {(["seen", "unseen"] as const).map((s) => Object.keys(sv[s].tasks).map((t) => (
          <span key={t} tabIndex={0} onPointerEnter={() => setHl(t)} onPointerLeave={() => setHl(null)} onFocus={() => setHl(t)} onBlur={() => setHl(null)}
            style={{ color: s === "seen" ? "var(--real)" : "var(--dream)", cursor: "default", fontWeight: hl === t ? 600 : 400 }}>
            {pretty(t)} <span className="muted">{sv[s].tasks[t].median ?? ">" + (dreams[t]?.T ?? 96)}</span>
          </span>
        )))}
      </div>
      <figcaption className="note">
        Numbers after each name are the task's median trust horizon in steps. Pooled medians: {sv.seen.pooled.median} steps on familiar tasks, {sv.unseen.pooled.median} step on unseen ones. Familiar test split has 8 dreams per task, unseen 16, so single-task curves are rough.
      </figcaption>
    </figure>
  );
}

/* ------------------------------------------------------------------ Phase 3 distance */

export function DistanceSurvival() {
  const bins = site.phase3.bins as { dist_range: [number, number]; n: number; frac_failed: number; median_trust_horizon: number | null; S: number[] }[];
  const ramp = ["#9fc7c1", "#5aa59b", "#1f7f74", "#0d4f48"];
  const rampDark = ["#3b6e68", "#4f978d", "#6fc0b4", "#a7e3da"];
  const dark = true;
  const names = ["nearest", "near", "far", "farthest"];
  const series = bins.map((b, i) => ({
    key: "q" + i, label: `${names[i]} (${b.dist_range[0].toFixed(2)}-${b.dist_range[1].toFixed(2)})`,
    values: [1, ...b.S], xs: Array.from({ length: b.S.length + 1 }, (_, j) => j), color: (dark ? rampDark : ramp)[i], width: 2.25, step: true,
  }));
  return (
    <figure className="fig">
      <p className="chart-title">Does a less familiar start state break sooner?</p>
      <p className="chart-sub">320 dreams on 5 familiar tasks, split into quarters by distance from the training data (1.0 = as far as a typical new expert episode starts). Darker = farther.</p>
      <LineChart ariaLabel="Survival curves by distance quartile" series={series} x0={0} xDomain={[0, 92]} yDomain={[0, 1]}
        xTicks={[0, 16, 32, 48, 64, 80]} yTicks={[0, 0.25, 0.5, 0.75, 1]} yFormat={(v) => `${v * 100}%`} xTitle="dream step"
        tipX={(v) => `step ${v}`} tipValue={(v) => `${Math.round(v * 100)}%`} refs={[{ y: 0.5 }]} height={260} />
      <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", fontFamily: "var(--sans)", fontSize: "var(--t-small)", marginTop: "0.5rem" }}>
        {series.map((s, i) => (
          <span key={s.key} className="key"><i style={{ background: s.color, height: 3 }} />{s.label}: median {bins[i].median_trust_horizon ?? "over 92"}</span>
        ))}
      </div>
      <figcaption className="note">
        The horizon does not fall steadily with distance. A statistical test of "farther starts break sooner" gives an effect too small to tell apart from zero: its range of plausible values (exploratory) runs from a little protective to clearly harmful. Most start states sat between 0.9 and 1.1, so this range was too narrow to show one.
      </figcaption>
    </figure>
  );
}

/* ------------------------------------------------------------------ unseen gallery */

export function Thumb({ task, kind, step }: { task: string; kind: "real" | "dream"; step: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let live = true;
    loadSprite(frameUrl(task, kind)).then((im) => { const c = ref.current?.getContext("2d"); if (live && c) drawFrame(c, im, step); });
    return () => { live = false; };
  }, [task, kind, step]);
  return <canvas ref={ref} width={128} height={128} aria-label={`${kind} frame of ${task} at step ${step}`} />;
}

export function UnseenGallery({ onPick }: { onPick: (t: string) => void }) {
  const [step, setStep] = useState(1);
  return (
    <div className="wide"><figure className="fig">
      <div style={{ display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap", marginBottom: "1rem" }}>
        <p className="chart-title" style={{ margin: 0 }}>The 10 unseen tasks, real (left) and dreamed (right)</p>
        <div className="seg" role="group" aria-label="Dream step to show">
          {[1, 8, 32, 64].map((s) => <button key={s} aria-pressed={step === s} onClick={() => setStep(s)}>step {s}</button>)}
        </div>
      </div>
      <div className="gallery">
        {unseenTasks.map((t) => {
          const d = dreams[t]; const s = Math.min(step, d.T);
          return (
            <button key={t} onClick={() => onPick(t)} aria-label={`Play the ${pretty(t)} dream`}>
              <div className="pair"><Thumb task={t} kind="real" step={s} /><Thumb task={t} kind="dream" step={s} /></div>
              <div className="name">{pretty(t)}</div>
              <div className="meta">{d.onset == null ? "this dream did not break" : d.onset === 1 ? "broken from step 1" : `broke at step ${d.onset}`}</div>
            </button>
          );
        })}
      </div>
      <figcaption className="note">Each pair is the dream shown in the player; the line under the name says when that dream broke. Click a task to play it at the top of the page. Look at cup catch var1 and finger turn easy var1: the dream paints the new scene in the colours of the familiar version of the task.</figcaption>
    </figure></div>
  );
}

/* ------------------------------------------------------------------ filmstrip */

const FILM_TASKS = ["walker-walk", "pygame-pong", "atari-ms-pacman", "og-point-var1"];

export function Filmstrip() {
  const [task, setTask] = useState(FILM_TASKS[0]);
  const d = dreams[task];
  const steps = [1, 8, 16, 32, 64, 96].map((s) => Math.min(s, d.T));
  return (
    <figure className="fig">
      <p className="chart-title">A photocopy of a photocopy</p>
      <p className="chart-sub">The same moments, real and dreamed. An orange outline marks steps after the dream broke.</p>
      <div className="seg" role="group" aria-label="Task" style={{ marginBottom: "0.8rem" }}>
        {FILM_TASKS.map((t) => (
          <button key={t} aria-pressed={task === t} onClick={() => setTask(t)}>
            {pretty(t)}{dreams[t].split === "unseen" ? " (unseen)" : ""}
          </button>
        ))}
      </div>
      <div className="filmstrip">
        <span />
        {steps.map((s) => <span key={"l" + s} className="steplab">step {s}</span>)}
        <span className="rowlab r">Real</span>
        {steps.map((s) => <div key={"r" + s}><Thumb task={task} kind="real" step={s} /></div>)}
        <span className="rowlab d">Dreamed</span>
        {steps.map((s) => <div key={"d" + s} className={d.onset != null && s >= d.onset ? "broken" : ""}><Thumb task={task} kind="dream" step={s} /></div>)}
      </div>
      <figcaption className="note">
        {d.onset == null ? "This dream never crossed the break line." : `This dream breaks at step ${d.onset}.`} The model saw 8 real frames before step 1 and the real actions throughout; everything in the bottom row is imagined.
      </figcaption>
    </figure>
  );
}
