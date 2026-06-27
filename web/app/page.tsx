import Link from "next/link";
import { data } from "./data";
import Nav from "../components/Nav";
import Footer from "../components/Footer";
import Reveal from "../components/Reveal";
import Counter from "../components/Counter";
import FidBars from "../components/FidBars";

function Cta({ href, children, dark }: { href: string; children: React.ReactNode; dark?: boolean }) {
  return (
    <Link
      href={href}
      className={
        dark
          ? "rounded-xl bg-white px-5 py-2.5 text-sm font-semibold text-slate-900 transition hover:bg-slate-100"
          : "rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-700"
      }
    >
      {children}
    </Link>
  );
}

const SAMPLES = [
  { kind: "walls", title: "Walls", note: "the given load-bearing structure", img: "/data/samples/10298_walls.png" },
  { kind: "rooms", title: "Rooms", note: "the target layout to generate", img: "/data/samples/10298_rooms.png" },
  { kind: "graph", title: "Access graph", note: "which rooms connect to which", img: "/data/samples/10298_graph.png" },
];

export default function Home() {
  const s = data.stats;
  return (
    <div className="h-[100dvh] snap-y snap-proximity overflow-y-scroll scroll-smooth bg-paper">
      <Nav />

      {/* ───────────────────── Slide 1 · Title ───────────────────── */}
      <section className="slide-dark relative flex min-h-[100dvh] snap-start items-center px-5 text-white">
        <div className="mx-auto w-full max-w-5xl py-24">
          <Reveal variant="fade">
            <span className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/5 px-3 py-1 text-xs font-medium text-indigo-200">
              <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
              Modified Swiss Dwellings · live on an AMD MI300X
            </span>
          </Reveal>
          <Reveal delay={120}>
            <h1 className="mt-6 text-5xl font-semibold leading-[1.05] tracking-tight md:text-7xl">
              Draw it. Train it.
              <br />
              <span className="bg-gradient-to-r from-indigo-300 via-sky-300 to-emerald-300 bg-clip-text text-transparent">
                Generate the floor plan.
              </span>
            </h1>
          </Reveal>
          <Reveal delay={260}>
            <p className="mt-6 max-w-2xl text-lg text-slate-300 md:text-xl">
              An end-to-end tool for the floor-plan generation challenge — sketch a structure and the model
              lays out the rooms, train new models on the GPU, and browse {s.apartments.toLocaleString()} real
              apartments. All from the browser.
            </p>
          </Reveal>
          <Reveal delay={420}>
            <div className="mt-9 flex flex-wrap gap-3">
              <Cta href="/studio" dark>
                ✦ Open the Studio
              </Cta>
              <Link
                href="/live"
                className="rounded-xl border border-white/25 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-white/10"
              >
                Watch it train live
              </Link>
            </div>
          </Reveal>
        </div>
        <div className="scroll-hint absolute bottom-8 left-1/2 -translate-x-1/2 text-sm text-slate-400">
          ↓ scroll the deck
        </div>
      </section>

      {/* ───────────────────── Slide 2 · The challenge ───────────────────── */}
      <section className="flex min-h-[100dvh] snap-start items-center px-5">
        <div className="mx-auto w-full max-w-6xl py-24">
          <Reveal>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-indigo-600">The challenge</p>
            <h2 className="mt-3 max-w-3xl text-4xl font-semibold tracking-tight text-slate-900 md:text-5xl">
              Given the walls, design the rooms.
            </h2>
          </Reveal>
          <Reveal delay={120}>
            <p className="mt-5 max-w-2xl text-lg text-slate-600">
              The structure only gives the load-bearing envelope — not the partition walls. The task: generate
              a complete, plausible room layout for every building.
            </p>
          </Reveal>
          <div className="mt-14 grid grid-cols-2 gap-5 md:grid-cols-4">
            {[
              { v: <Counter to={s.apartments} />, l: "apartments" },
              { v: <Counter to={s.geometries / 1e6} decimals={2} suffix="M" />, l: "geometries" },
              { v: <Counter to={s.trainSamples} />, l: "train plans" },
              { v: <Counter to={s.testSamples} />, l: "test plans" },
            ].map((m, i) => (
              <Reveal key={i} variant="up" delay={i * 110}>
                <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                  <div className="text-4xl font-semibold tracking-tight text-slate-900 md:text-5xl">{m.v}</div>
                  <div className="mt-2 text-sm font-medium uppercase tracking-wide text-slate-500">{m.l}</div>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ───────────────────── Slide 3 · The data ───────────────────── */}
      <section className="flex min-h-[100dvh] snap-start items-center border-y border-slate-200 bg-white px-5">
        <div className="mx-auto w-full max-w-6xl py-24">
          <Reveal>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-indigo-600">What the model sees</p>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight text-slate-900 md:text-5xl">
              Three views of every home.
            </h2>
          </Reveal>
          <div className="mt-14 grid gap-6 md:grid-cols-3">
            {SAMPLES.map((sm, i) => (
              <Reveal key={sm.kind} variant="up" delay={i * 130}>
                <figure className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 shadow-sm">
                  <div className="aspect-square w-full bg-white">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={sm.img} alt={sm.title} className="h-full w-full object-contain" />
                  </div>
                  <figcaption className="border-t border-slate-200 px-4 py-3">
                    <div className="text-base font-semibold text-slate-900">{sm.title}</div>
                    <div className="text-sm text-slate-500">{sm.note}</div>
                  </figcaption>
                </figure>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ───────────────────── Slide 4 · Models / leaderboard ───────────────────── */}
      <section className="flex min-h-[100dvh] snap-start items-center px-5">
        <div className="mx-auto grid w-full max-w-6xl items-center gap-12 py-24 lg:grid-cols-2">
          <Reveal variant="left">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-indigo-600">What we built</p>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight text-slate-900 md:text-5xl">
              A leaderboard of models.
            </h2>
            <p className="mt-5 max-w-md text-lg text-slate-600">
              From a learned U-Net to rule-based generators that exploit how real homes are shaped —
              <strong> rectangular rooms aligned to the outer walls</strong>, balconies pushed to the facade.
              Each one is documented and scored live.
            </p>
            <div className="mt-7">
              <Cta href="/models">See every model →</Cta>
            </div>
          </Reveal>
          <Reveal variant="right" delay={140}>
            <div className="rounded-2xl border border-slate-200 bg-white p-7 shadow-sm">
              <FidBars />
            </div>
          </Reveal>
        </div>
      </section>

      {/* ───────────────────── Slide 5 · Studio ───────────────────── */}
      <section className="slide-dark flex min-h-[100dvh] snap-start items-center px-5 text-white">
        <div className="mx-auto w-full max-w-5xl py-24">
          <Reveal>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-indigo-300">Studio</p>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight md:text-6xl">
              Draw a structure,
              <br />
              get an apartment.
            </h2>
          </Reveal>
          <Reveal delay={150}>
            <p className="mt-6 max-w-2xl text-lg text-slate-300">
              Sketch the walls with your pencil — they snap straight and clean. Pick a model, hit generate,
              and watch the rooms get laid out inside, right where you drew them.
            </p>
          </Reveal>
          <Reveal delay={300}>
            <div className="mt-9">
              <Cta href="/studio" dark>
                ✦ Open the Studio
              </Cta>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ───────────────────── Slide 6 · Live ───────────────────── */}
      <section className="flex min-h-[100dvh] snap-start items-center border-y border-slate-200 bg-white px-5">
        <div className="mx-auto w-full max-w-5xl py-24">
          <Reveal>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-indigo-600">Live</p>
            <h2 className="mt-3 text-4xl font-semibold tracking-tight text-slate-900 md:text-6xl">
              Train on the GPU.
              <br />
              Watch it learn.
            </h2>
          </Reveal>
          <Reveal delay={150}>
            <p className="mt-6 max-w-2xl text-lg text-slate-600">
              Launch a real training run from the browser, watch the loss fall live, then compare every
              generated plan with the real ground truth — side by side.
            </p>
          </Reveal>
          <Reveal delay={300}>
            <div className="mt-9 flex flex-wrap gap-3">
              <Cta href="/live">Open the Live page →</Cta>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ───────────────────── Slide 7 · Closing ───────────────────── */}
      <section className="flex min-h-[100dvh] snap-start items-center px-5">
        <div className="mx-auto w-full max-w-5xl py-24 text-center">
          <Reveal variant="scale">
            <h2 className="text-4xl font-semibold tracking-tight text-slate-900 md:text-6xl">
              Explore it yourself.
            </h2>
          </Reveal>
          <Reveal delay={150}>
            <div className="mx-auto mt-10 grid max-w-3xl grid-cols-2 gap-4 md:grid-cols-4">
              {[
                ["/studio", "Studio", "draw → generate"],
                ["/live", "Live", "train & compare"],
                ["/models", "Models", "the leaderboard"],
                ["/research", "Research", "ideas & papers"],
              ].map(([href, t, d]) => (
                <Link
                  key={href}
                  href={href}
                  className="group rounded-2xl border border-slate-200 bg-white p-5 text-left shadow-sm transition hover:border-indigo-300 hover:shadow"
                >
                  <div className="text-lg font-semibold text-slate-900">{t}</div>
                  <div className="mt-1 text-sm text-slate-500">{d}</div>
                  <span className="mt-3 inline-block text-sm font-medium text-indigo-600 group-hover:underline">
                    Open →
                  </span>
                </Link>
              ))}
            </div>
          </Reveal>
        </div>
      </section>

      <Footer />
    </div>
  );
}
