import { useEffect, useRef, useState } from "react";

/** Width of a container in CSS pixels, so SVG text is drawn at its true size on any screen. */
export function useWidth<T extends HTMLElement>(fallback: number) {
  const ref = useRef<T>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current; if (!el) return;
    const ro = new ResizeObserver(([e]) => { const x = Math.round(e.contentRect.width); if (x > 0) setW(x); });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/** For an SVG with a fixed viewBox width: the font size, in its own units, that shows as 16 px on screen. */
export function useSvgFont<T extends HTMLElement>(viewBoxWidth: number, fallback = viewBoxWidth) {
  const [ref, w] = useWidth<T>(fallback);
  return [ref, (16 * viewBoxWidth) / Math.max(w, 1)] as const;
}
