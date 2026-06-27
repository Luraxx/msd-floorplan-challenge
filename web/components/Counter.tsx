"use client";

import { useEffect, useRef, useState } from "react";

export default function Counter({
  to,
  duration = 1500,
  decimals = 0,
  suffix = "",
  prefix = "",
}: {
  to: number;
  duration?: number;
  decimals?: number;
  suffix?: string;
  prefix?: string;
}) {
  const ref = useRef<HTMLSpanElement | null>(null);
  const started = useRef(false);
  const [v, setV] = useState(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting && !started.current) {
          started.current = true;
          io.disconnect();
          let raf = 0;
          let t0 = 0;
          const tick = (t: number) => {
            if (!t0) t0 = t;
            const p = Math.min(1, (t - t0) / duration);
            setV(to * (1 - Math.pow(1 - p, 3))); // ease-out cubic
            if (p < 1) raf = requestAnimationFrame(tick);
          };
          raf = requestAnimationFrame(tick);
          return () => cancelAnimationFrame(raf);
        }
      },
      { threshold: 0.5 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [to, duration]);

  const text =
    decimals > 0
      ? v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
      : Math.round(v).toLocaleString();

  return (
    <span ref={ref}>
      {prefix}
      {text}
      {suffix}
    </span>
  );
}
