"use client";

import { useEffect, useRef, useState } from "react";

type Variant = "up" | "down" | "left" | "right" | "scale" | "fade";

const HIDDEN: Record<Variant, string> = {
  up: "opacity-0 translate-y-10",
  down: "opacity-0 -translate-y-10",
  left: "opacity-0 -translate-x-12",
  right: "opacity-0 translate-x-12",
  scale: "opacity-0 scale-95",
  fade: "opacity-0",
};

export default function Reveal({
  children,
  className = "",
  variant = "up",
  delay = 0,
  once = true,
}: {
  children: React.ReactNode;
  className?: string;
  variant?: Variant;
  delay?: number;
  once?: boolean;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [vis, setVis] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setVis(true);
          if (once) io.disconnect();
        } else if (!once) {
          setVis(false);
        }
      },
      { threshold: 0.18, rootMargin: "0px 0px -8% 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [once]);

  return (
    <div
      ref={ref}
      style={{ transitionDelay: `${delay}ms` }}
      className={`transition-all duration-700 ease-out will-change-transform ${
        vis ? "translate-x-0 translate-y-0 scale-100 opacity-100" : HIDDEN[variant]
      } ${className}`}
    >
      {children}
    </div>
  );
}
