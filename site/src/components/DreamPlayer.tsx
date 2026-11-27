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
      <figcaption><b>{kind === "real" ? "Real" : "Dreamed"}</b></figcaption>
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
          Dream
          <select value={task} onChange={(e) => setTask(e.target.value)}>
            <optgroup label="Familiar tasks (held-out episodes)">
              {seenTasks.map((t) => <option key={t} value={t}>{pretty(t)}</option>)}
            </optgroup>
            <optgroup label="Unseen tasks (never trained on)">
              {unseenTasks.map((t) => <option key={t} value={t}>{pretty(t)}</option>)}
            </optgroup>
          </select>
        </label>
        <span className="muted" style={{ fontSize: "var(--t-small)" }}>
          {d.split === "seen" ? "Familiar task" : "Unseen task"} · after 8 real frames, the model dreams {d.T} steps on its own
        </span>
      </div>

      <div className="player-body">
        <div>
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
              : <span className="state ok">Not broken yet</span>}
            <span className="muted" style={{ fontSize: "var(--t-small)" }}>
              {d.onset == null ? "This dream never broke within its horizon." : !broken ? `It will break at step ${d.onset}.` : ""}
            </span>
          </div>
          <div className="readouts">
            <div className="readout">
              <div className="v" style={{ color: broken ? "var(--ink-2)" : undefined }}>{broken ? "done" : pct(d.p_forecast[i])}</div>
              <div className="k"><span className="sw solid" />Forecast: chance a sound dream breaks within the next 8 steps</div>
            </div>
            <div className="readout">
              <div className="v">{pct(d.p_detect[i])}</div>
              <div className="k"><span className="sw dashed" />Alarm: chance the dream has already broken</div>
            </div>
          </div>
          <p className="muted" style={{ fontSize: "var(--t-small)", lineHeight: 1.45, margin: "0.75rem 0 0" }}>
            Two different questions. The forecast only makes sense while the dream is still sound, so it stops at the break. The alarm asks whether it has broken by now; it is meant to be low before the break and high after. A good forecast would climb <em>before</em> the break. Watch whether it does.
          </p>
        </div>

        <div className="panels">
          <p className="panel-title">
            <span><b style={{ color: "var(--ink)" }}>How wrong the dream is</b>, as a share of the distance between two random moments of this task. It breaks once it stays above half.</span>
          </p>
          <LineChart ariaLabel="Latent error of the dream over time, with the break line at half the chance error"
            series={[{ key: "err", label: "error", values: d.err, color: "var(--ink)", width: 2 }]}
            xDomain={[1, d.T]} yDomain={[0, errMax]} xTicks={xTicks} yTicks={[0, 0.5, 1, 1.5, 2, 2.5, 3].filter((t) => t <= errMax)}
            refs={[{ y: 0.5, label: "break line", color: "var(--broken)", anchor: "start" },
              ...(d.onset ? [{ x: d.onset, label: `breaks at ${d.onset}`, color: "var(--broken)", dash: "2 3" }] : [])]}
            shadeFrom={d.onset} cursor={step} onHover={onHover} height={170} tipX={(x) => `step ${x}`}
            yFormat={(v) => `${v}x`} tipValue={(v) => `${v.toFixed(2)}x chance`} />

          <p className="panel-title">
            <span><b style={{ color: "var(--ink)" }}>The model's probabilities</b>, using only signals up to that step</span>
            <span className="key"><i className="solid" />forecast: breaks within 8 steps</span>
            <span className="key"><i className="dashed" />alarm: already broken</span>
          </p>
          <LineChart ariaLabel="Forecast and detection probabilities over the dream"
            series={[
              { key: "pf", label: "forecast", values: d.p_forecast.map((v, j) => (d.onset != null && j + 1 > d.onset ? null : v)), color: "var(--dream)", width: 2.5 },
              { key: "pd", label: "alarm", values: d.p_detect, color: "var(--real)", width: 2.5, dash: "7 5" },
            ]}
            xDomain={[1, d.T]} yDomain={[0, 1]} xTicks={xTicks} yTicks={[0, 0.5, 1]} yFormat={(v) => pct(v)}
            shadeFrom={d.onset} cursor={step} onHover={onHover} height={150} tipX={(x) => `step ${x}`} tipValue={pct} />

          <p className="panel-title">
            <span><b style={{ color: "var(--ink)" }}>The three warning signals</b>, per unit of scene motion, log scale</span>
            <span className="key"><i style={{ background: "var(--s-ur)" }} /><span>u<sub>r</sub> round trip</span></span>
            <span className="key"><i style={{ background: "var(--s-uf)" }} /><span>u<sub>f</sub> flow</span></span>
            <span className="key"><i style={{ background: "var(--s-us)" }} /><span>u<sub>s</sub> seed spread</span></span>
          </p>
          <LineChart ariaLabel="The three warning signals over the dream, log scale"
            series={[
              { key: "ur", label: "u_r", values: d.u_r, color: "var(--s-ur)", width: 1.5 },
              { key: "uf", label: "u_f", values: d.u_f, color: "var(--s-uf)", width: 1.5 },
              { key: "us", label: "u_s", values: d.u_s, color: "var(--s-us)", width: 1.5 },
            ]}
            xDomain={[1, d.T]} yDomain={[sMin, sMax]} yLog xTicks={xTicks} yTicks={sTicks} yFormat={(v) => String(v)}
            shadeFrom={d.onset} cursor={step} onHover={onHover} height={150} xTitle="dream step" tipX={(x) => `step ${x}`}
            tipValue={(v) => v.toPrecision(2)} />
        </div>
      </div>
    </div>
  );
}
