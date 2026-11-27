import { useEffect, useRef, useState } from "react";
import LineChart from "./LineChart";
import { dreams, drawFrame, frameUrl, loadSprite, pretty, seenTasks, unseenTasks } from "../data";

const reduceMotion = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

function Frame({ task, kind, step, broken }: { task: string; kind: "real" | "dream"; step: number; broken?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [im, setIm] = useState<HTMLImageElement | null>(null);
  useEffect(() => { let live = true; loadSprite(frameUrl(task, kind)).then((x) => live && setIm(x)); return () => { live = false; }; }, [task, kind]);
  useEffect(() => { const c = ref.current?.getContext("2d"); if (c && im) drawFrame(c, im, step); }, [im, step]);
  return (
    <figure className={`frame ${kind}${broken ? " broken" : ""}`} style={{ margin: 0 }}>
      <figcaption><b>{kind === "real" ? "What really happened" : "What the world model imagined"}</b></figcaption>
      <canvas ref={ref} width={256} height={256} aria-label={`${kind} frame at step ${step}`} />
    </figure>
  );
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

export default function DreamPlayer({ task, setTask }: { task: string; setTask: (t: string) => void }) {
  const d = dreams[task];
  const [step, setStep] = useState(1);
  const [playing, setPlaying] = useState(!reduceMotion);
  const [fps, setFps] = useState(8);
  const hovering = useRef(false);

  useEffect(() => { setStep(1); }, [task]);
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      if (hovering.current) return;
      setStep((s) => (s >= d.T ? 1 : s + 1));
    }, 1000 / fps);
    return () => window.clearInterval(id);
  }, [playing, fps, d.T]);

  const i = step - 1;
  const broken = d.onset != null && step >= d.onset;
  const xTicks = [1, 16, 32, 48, 64, 80, 96].filter((t) => t <= d.T);
  const onHover = (x: number | null) => { hovering.current = x != null; if (x != null) setStep(x); };
  const errMax = Math.max(1.2, Math.ceil(Math.max(...d.err) * 2) / 2);
  const sigVals = [...d.u_r, ...d.u_f, ...d.u_s].filter((v) => v > 0);
  const sMin = Math.pow(10, Math.floor(Math.log10(Math.min(...sigVals))));
  const sMax = Math.pow(10, Math.ceil(Math.log10(Math.max(...sigVals))));
  const sTicks = [0.001, 0.01, 0.1, 1, 10, 100].filter((t) => t >= sMin && t <= sMax);

  return (
    <div className="player" id="player">
      <div className="player-head">
        <label>
          Pick a dream
          <select value={task} onChange={(e) => setTask(e.target.value)}>
            <optgroup label="Familiar tasks: the world model trained on these">
              {seenTasks.map((t) => <option key={t} value={t}>{pretty(t)}</option>)}
            </optgroup>
            <optgroup label="Unseen tasks: the world model never saw these">
              {unseenTasks.map((t) => <option key={t} value={t}>{pretty(t)}</option>)}
            </optgroup>
          </select>
        </label>
        <span className="muted" style={{ fontSize: "var(--t-small)" }}>
          {d.split === "seen"
            ? "A familiar task. The world model trained on this task, but never on this episode."
            : "An unseen task. The world model never saw this task during training."}
        </span>
      </div>

      <ol className="howto">
        <li><span className="badge">1</span><span>The world model was shown 8 real frames and the actions, then imagined the next {d.T} frames on its own, one step at a time. Compare what it imagined with what really happened.</span></li>
        <li><span className="badge">2</span><span>We measure how far apart the two are. When the gap stays above the red line, we say the dream has <b>broken</b>: what it imagines no longer matches reality.</span></li>
        <li><span className="badge">3</span><span>Our two predictions try to tell, without ever seeing the real frames, whether the dream is about to break or already has.</span></li>
        <li><span className="badge">4</span><span>They are built only from the world model's own uncertainty signals.</span></li>
      </ol>

      <div className="player-body">
        <div>
          <p className="panel-title ptitle"><span className="badge">1</span><span><b style={{ color: "var(--ink)" }}>Real against imagined</b></span></p>
          <div className="frames">
            <Frame task={task} kind="real" step={step} />
            <Frame task={task} kind="dream" step={step} broken={broken} />
          </div>
          <div className="transport">
            <button onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause" : "Play"}>{playing ? "Pause" : "Play"}</button>
            <input type="range" min={1} max={d.T} value={step} aria-label="Dream step"
              onChange={(e) => { setPlaying(false); setStep(+e.target.value); }} />
            <span className="stepnum">step {step}</span>
          </div>
          <div className="transport" style={{ marginTop: "0.4rem" }}>
            <span className="muted" style={{ fontSize: "var(--t-small)" }}>Speed</span>
            <div className="seg" role="group" aria-label="Playback speed">
              {[4, 8, 16].map((f) => <button key={f} aria-pressed={fps === f} onClick={() => setFps(f)}>{f / 8}x</button>)}
            </div>
          </div>
          <div className="status" aria-live="polite">
            {broken
              ? <span className="state bad">Broken since step {d.onset}</span>
              : <span className="state ok">Still matches reality</span>}
            <span className="muted" style={{ fontSize: "var(--t-small)" }}>
              {d.onset == null ? "This dream never broke." : !broken ? `It breaks at step ${d.onset}.` : "The imagined frames no longer match what happened."}
            </span>
          </div>
          <div className="readouts">
            <div className="readout">
              <div className="v" style={{ color: broken ? "var(--ink-2)" : undefined, fontSize: broken ? "1.1rem" : undefined }}>{broken ? "already broke" : pct(d.p_forecast[i])}</div>
              <div className="k"><span className="sw solid" /><b>Forecast:</b> chance this dream breaks within the next 8 steps</div>
            </div>
            <div className="readout">
              <div className="v">{pct(d.p_detect[i])}</div>
              <div className="k"><span className="sw dashed" /><b>Alarm:</b> chance this dream has already broken</div>
            </div>
          </div>
          <p className="muted" style={{ fontSize: "var(--t-small)", lineHeight: 1.45, margin: "0.75rem 0 0" }}>
            How to judge them. A useful forecast climbs <em>before</em> the break; it stops at the break, because a broken dream cannot break again. A useful alarm stays low before the break and turns high after it.
          </p>
        </div>

        <div className="panels">
          <p className="panel-title ptitle"><span className="badge">2</span><span><b style={{ color: "var(--ink)" }}>How far the imagined frames are from the real ones</b>. 1x means as far apart as two random moments of this task. Above the red line for 5 steps in a row means broken.</span></p>
          <LineChart ariaLabel="Gap between imagined and real frames over time, with the break line at half the gap between two random moments"
            series={[{ key: "err", label: "gap", values: d.err, color: "var(--ink)", width: 2 }]}
            xDomain={[1, d.T]} yDomain={[0, errMax]} xTicks={xTicks} yTicks={[0, 0.5, 1, 1.5, 2, 2.5, 3].filter((t) => t <= errMax)}
            refs={[{ y: 0.5, label: "break line (0.5x)", color: "var(--broken)", anchor: "start" },
              ...(d.onset ? [{ x: d.onset, label: `breaks at ${d.onset}`, color: "var(--broken)", dash: "2 3" }] : [])]}
            shadeFrom={d.onset} cursor={step} onHover={onHover} height={170} tipX={(x) => `step ${x}`}
            yFormat={(v) => `${v}x`} tipValue={(v) => `${v.toFixed(2)}x`} />

          <p className="panel-title ptitle"><span className="badge">3</span><span><b style={{ color: "var(--ink)" }}>Our two predictions</b>, made at each step without seeing the real frames</span></p>
          <p className="panel-title" style={{ marginTop: 0 }}>
            <span className="key"><i className="solid" />Forecast: breaks within the next 8 steps</span>
            <span className="key"><i className="dashed" />Alarm: has already broken</span>
          </p>
          <LineChart ariaLabel="Forecast and alarm probabilities over the dream"
            series={[
              { key: "pf", label: "Forecast", values: d.p_forecast.map((v, j) => (d.onset != null && j + 1 > d.onset ? null : v)), color: "var(--dream)", width: 2.5 },
              { key: "pd", label: "Alarm", values: d.p_detect, color: "var(--real)", width: 2.5, dash: "7 5" },
            ]}
            xDomain={[1, d.T]} yDomain={[0, 1]} xTicks={xTicks} yTicks={[0, 0.5, 1]} yFormat={(v) => pct(v)}
            shadeFrom={d.onset} cursor={step} onHover={onHover} height={150} tipX={(x) => `step ${x}`} tipValue={pct} />

          <p className="panel-title ptitle"><span className="badge">4</span><span><b style={{ color: "var(--ink)" }}>The world model's uncertainty signals</b>: higher means less sure. Scaled by how much the scene moves; log scale.</span></p>
          <p className="panel-title" style={{ marginTop: 0 }}>
            <span className="key"><i style={{ background: "var(--s-ur)" }} /><span>round trip (u<sub>r</sub>)</span></span>
            <span className="key"><i style={{ background: "var(--s-uf)" }} /><span>second thoughts (u<sub>f</sub>)</span></span>
            <span className="key"><i style={{ background: "var(--s-us)" }} /><span>two tries disagree (u<sub>s</sub>)</span></span>
          </p>
          <LineChart ariaLabel="The world model's three uncertainty signals over the dream, log scale"
            series={[
              { key: "ur", label: "round trip", values: d.u_r, color: "var(--s-ur)", width: 1.5 },
              { key: "uf", label: "second thoughts", values: d.u_f, color: "var(--s-uf)", width: 1.5 },
              { key: "us", label: "two tries disagree", values: d.u_s, color: "var(--s-us)", width: 1.5 },
            ]}
            xDomain={[1, d.T]} yDomain={[sMin, sMax]} yLog xTicks={xTicks} yTicks={sTicks} yFormat={(v) => String(v)}
            shadeFrom={d.onset} cursor={step} onHover={onHover} height={150} xTitle="dream step" tipX={(x) => `step ${x}`}
            tipValue={(v) => v.toPrecision(2)} />
        </div>
      </div>
    </div>
  );
}
