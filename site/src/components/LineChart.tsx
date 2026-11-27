import { useMemo, useRef, useState } from "react";
import { useWidth } from "./useWidth";

export interface Series {
  key: string;
  label: string;
  xs?: number[];               // default: x0, x0 + 1, ...
  values: (number | null)[];
  color: string;
  width?: number;
  dash?: string;
  opacity?: number;
  step?: boolean;              // step-after curve (survival)
  band?: { lo: number[]; hi: number[] };
  endLabel?: boolean;          // direct label at the right end
  quiet?: boolean;             // excluded from the tooltip
}

export interface RefLine { y?: number; x?: number; label?: string; color?: string; dash?: string; anchor?: "start" | "end" }

interface Props {
  series: Series[];
  x0?: number;
  xDomain: [number, number];
  yDomain: [number, number];
  yLog?: boolean;
  xTicks: number[];
  yTicks: number[];
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  xTitle?: string;
  yTitle?: string;
  height?: number;
  width?: number;
  refs?: RefLine[];
  shadeFrom?: number | null;   // x from which the plot is washed in the "broken" colour
  cursor?: number | null;
  onHover?: (x: number | null) => void;
  tipX?: (x: number) => string;
  tipValue?: (v: number) => string;
  rightPad?: number;
  ariaLabel: string;
}

export default function LineChart(p: Props) {
  const [box, bw] = useWidth<HTMLDivElement>(p.width ?? 640);
  const W = Math.min(bw, p.width ?? 10000), H = p.height ?? 220;
  const m = { l: 44, r: W < 420 ? Math.min(p.rightPad ?? 16, 70) : (p.rightPad ?? 16), t: 10, b: p.xTitle ? 40 : 24 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const ref = useRef<SVGSVGElement>(null);
  const [hx, setHx] = useState<number | null>(null);
  const x0 = p.x0 ?? 1;

  const X = (v: number) => m.l + ((v - p.xDomain[0]) / (p.xDomain[1] - p.xDomain[0])) * iw;
  const ly = (v: number) => (p.yLog ? Math.log10(Math.max(v, 1e-9)) : v);
  const [ya, yb] = [ly(p.yDomain[0]), ly(p.yDomain[1])];
  const Y = (v: number) => m.t + ih - ((Math.min(Math.max(ly(v), ya), yb) - ya) / (yb - ya)) * ih;

  const paths = useMemo(() => p.series.map((s) => {
    const xs = s.xs ?? s.values.map((_, i) => x0 + i);
    let d = "", pen = false, prevY = 0;
    s.values.forEach((v, i) => {
      if (v == null || !isFinite(v)) { pen = false; return; }
      const x = X(xs[i]), y = Y(v);
      if (!pen) d += `M${x.toFixed(1)} ${y.toFixed(1)}`;
      else if (s.step) d += `H${x.toFixed(1)}V${y.toFixed(1)}`;
      else d += `L${x.toFixed(1)} ${y.toFixed(1)}`;
      pen = true; prevY = y;
    });
    void prevY;
    let band = "";
    if (s.band) {
      const top = s.band.hi.map((v, i) => `${i ? "L" : "M"}${X(xs[i]).toFixed(1)} ${Y(v).toFixed(1)}`).join("");
      const bot = s.band.lo.map((v, i) => `L${X(xs[i]).toFixed(1)} ${Y(v).toFixed(1)}`).reverse().join("");
      band = top + bot + "Z";
    }
    return { s, xs, d, band };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [p.series, p.xDomain[0], p.xDomain[1], p.yDomain[0], p.yDomain[1], W, H]);

  const cur = p.cursor ?? hx;

  function move(e: React.PointerEvent) {
    const svg = ref.current; if (!svg) return;
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const v = p.xDomain[0] + ((px - m.l) / iw) * (p.xDomain[1] - p.xDomain[0]);
    const xsAll = paths[0]?.xs ?? [];
    let best = xsAll[0], bd = Infinity;
    for (const x of xsAll) { const d = Math.abs(x - v); if (d < bd) { bd = d; best = x; } }
    if (best == null) return;
    setHx(best); p.onHover?.(best);
  }
  function leave() { setHx(null); p.onHover?.(null); }

  // tooltip content at the hovered x
  let tip: { x: number; y: number; lines: { c: string; t: string }[] } | null = null;
  if (hx != null) {
    const lines: { c: string; t: string }[] = [];
    let ymin = Infinity;
    for (const { s, xs } of paths) {
      if (s.quiet) continue;
      const i = xs.indexOf(hx);
      const v = i >= 0 ? s.values[i] : null;
      if (v == null) continue;
      ymin = Math.min(ymin, Y(v));
      lines.push({ c: s.color, t: `${s.label}: ${p.tipValue ? p.tipValue(v) : v.toFixed(2)}` });
    }
    if (lines.length) tip = { x: X(hx), y: Math.max(ymin, m.t + 8), lines };
  }

  const fx = p.xFormat ?? ((v: number) => String(v));
  const fy = p.yFormat ?? ((v: number) => String(v));

  return (
    <div className="chart" ref={box} style={{ position: "relative" }}>
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={p.ariaLabel}
        onPointerMove={move} onPointerLeave={leave} style={{ touchAction: "pan-y" }}>
        {p.shadeFrom != null && (
          <rect x={X(p.shadeFrom) } y={m.t} width={Math.max(0, X(p.xDomain[1]) - X(p.shadeFrom))} height={ih} fill="var(--broken-wash)" />
        )}
        <g className="grid">
          {p.yTicks.map((t) => <line key={t} x1={m.l} x2={m.l + iw} y1={Y(t)} y2={Y(t)} />)}
        </g>
        <g className="axis">
          {p.yTicks.map((t) => <text key={t} x={m.l - 6} y={Y(t) + 4} textAnchor="end">{fy(t)}</text>)}
          {p.xTicks.map((t) => <text key={t} x={X(t)} y={m.t + ih + 16} textAnchor="middle">{fx(t)}</text>)}
          {p.xTitle && <text className="axis-title" x={m.l + iw / 2} y={H - 4} textAnchor="middle">{p.xTitle}</text>}
          {p.yTitle && <text className="axis-title" x={m.l} y={m.t - 2} textAnchor="start" dy="-0.2em">{p.yTitle}</text>}
        </g>
        <line className="baseline" x1={m.l} x2={m.l + iw} y1={m.t + ih} y2={m.t + ih} />
        {(p.refs ?? []).map((r, i) => r.y != null ? (
          <g key={i}>
            <line x1={m.l} x2={m.l + iw} y1={Y(r.y)} y2={Y(r.y)} stroke={r.color ?? "var(--ink-2)"} strokeDasharray={r.dash ?? "4 4"} strokeWidth={1.25} />
            {r.label && <text className="label-2" x={r.anchor === "start" ? m.l + 6 : m.l + iw - 2} y={Y(r.y) - 5} textAnchor={r.anchor ?? "end"}>{r.label}</text>}
          </g>
        ) : (
          <g key={i}>
            <line x1={X(r.x!)} x2={X(r.x!)} y1={m.t} y2={m.t + ih} stroke={r.color ?? "var(--ink-2)"} strokeDasharray={r.dash ?? "4 4"} strokeWidth={1.25} />
            {r.label && <text className="label-2" x={X(r.x!) + 5} y={m.t + 11}>{r.label}</text>}
          </g>
        ))}
        {paths.map(({ s, band }) => band && <path key={s.key + "b"} d={band} fill={s.color} opacity={0.14} />)}
        {paths.map(({ s, d }) => (
          <path key={s.key} d={d} fill="none" stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dash}
            strokeLinejoin="round" strokeLinecap="round" opacity={s.opacity ?? 1} />
        ))}
        {paths.map(({ s, xs }) => {
          if (!s.endLabel) return null;
          let i = s.values.length - 1; while (i > 0 && s.values[i] == null) i--;
          const v = s.values[i]; if (v == null) return null;
          return <text key={s.key + "l"} className="label" x={X(xs[i]) + 6} y={Y(v) + 4}>{s.label}</text>;
        })}
        {cur != null && cur >= p.xDomain[0] && cur <= p.xDomain[1] && (
          <line className="cursor" x1={X(cur)} x2={X(cur)} y1={m.t} y2={m.t + ih} />
        )}
        {hx != null && paths.map(({ s, xs }) => {
          if (s.quiet) return null;
          const i = xs.indexOf(hx); const v = i >= 0 ? s.values[i] : null;
          return v == null ? null : <circle key={s.key + "c"} cx={X(hx)} cy={Y(v)} r={4} fill={s.color} stroke="var(--paper)" strokeWidth={2} />;
        })}
        <rect x={m.l} y={m.t} width={iw} height={ih} fill="transparent" />
      </svg>
      {tip && (
        <div className="tooltip" style={{ left: `${(tip.x / W) * 100}%`, top: `${(tip.y / H) * 100}%` }}>
          <div style={{ opacity: 0.75 }}>{p.tipX ? p.tipX(hx!) : fx(hx!)}</div>
          {tip.lines.map((l, i) => <div key={i}><span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 4, background: l.c, marginRight: 6 }} />{l.t}</div>)}
        </div>
      )}
    </div>
  );
}
