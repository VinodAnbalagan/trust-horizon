import { useEffect, useState } from "react";
import DreamPlayer from "./components/DreamPlayer";
import LineChart from "./components/LineChart";
import {
  AlignedAtBreak, AurocViews, DistanceSurvival, Filmstrip, Reliability, SurvivalCurves, UnseenGallery, WarningHist, WindowSweep,
} from "./components/Figures";
import { dreams, pretty, site } from "./data";
import { ProbabilityCards } from "./components/Probability";
import { BrokenExplainer, DreamMachine, ThreeDreams } from "./components/Explainers";
import { DataTable, SignalsExplainer } from "./components/Explainers2";
import { AlignExplainer, PairsGame, PredictionExplainer, SurvivalLanes } from "./components/Explainers3";

const REPO = "https://github.com/VinodAnbalagan/trust-horizon";
const PAPER = "https://arxiv.org/abs/2606.27326";
const PAPER_PAGE = "https://www.nicklashansen.com/mmbench2";
const PAPER_CODE = "https://github.com/nicklashansen/mmbench2";

const SECTIONS = [
  { id: "dream", t: "How a world model dreams" },
  { id: "broken", t: "When is a dream broken?" },
  { id: "signals", t: "The three signals" },
  { id: "data", t: "The data" },
  { id: "predictions", t: "Our two predictions" },
  { id: "explore", t: "Explore every dream" },
  { id: "results", t: "Alarms, not forecasts" },
  { id: "horizon", t: "The trust horizon" },
  { id: "unseen", t: "Unseen tasks" },
  { id: "distance", t: "Distance from training" },
  { id: "meaning", t: "What it means" },
  { id: "probability", t: "The probability behind it" },
  { id: "honest", t: "How we checked" },
];

const PICKS: { t: string; why: string }[] = [
  { t: "cheetah-run", why: "slow drift" },
  { t: "pygame-pong", why: "breaks at step 4" },
  { t: "pygame-point-maze-var1", why: "never breaks" },
  { t: "cup-catch-var1", why: "unseen, broken at step 1" },
  { t: "pygame-foraging", why: "unseen, dissolves" },
];

function useActiveSection() {
  const [active, setActive] = useState<string>("");
  useEffect(() => {
    const els = SECTIONS.map((s) => document.getElementById(s.id)).filter(Boolean) as HTMLElement[];
    const io = new IntersectionObserver((entries) => {
      const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (vis[0]) setActive(vis[0].target.id);
    }, { rootMargin: "-20% 0px -70% 0px" });
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, []);
  return active;
}

function Chapter({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section className="chapter" id={id} aria-labelledby={id + "-h"}>
      <h2 id={id + "-h"}>{title}</h2>
      {children}
    </section>
  );
}

function Split({ text, fig }: { text: React.ReactNode; fig: React.ReactNode }) {
  return (
    <div className="split">
      <div className="text-col">{text}</div>
      <div className="fig-col">{fig}</div>
    </div>
  );
}

function BreakDefinition() {
  const d = dreams["cheetah-run"];
  return (
    <figure className="fig">
      <p className="chart-title">One dream, one break</p>
      <p className="chart-sub">Cheetah run: the dream's error as a share of the chance error. It breaks once the smoothed error stays above half.</p>
      <LineChart ariaLabel="Error of the cheetah run dream crossing the break line at step 37"
        series={[{ key: "err", label: "error", values: d.err, color: "var(--ink)" }]}
        xDomain={[1, d.T]} yDomain={[0, 1]} xTicks={[1, 16, 32, 48, 64, 80, 96]} yTicks={[0, 0.5, 1]}
        yFormat={(v) => `${v}x`} tipValue={(v) => `${v.toFixed(2)}x chance`} tipX={(x) => `step ${x}`}
        refs={[{ y: 0.5, label: "break line", color: "var(--broken)", anchor: "start" }, { x: d.onset!, label: `breaks at ${d.onset}`, color: "var(--broken)", dash: "2 3" }]}
        shadeFrom={d.onset} xTitle="dream step" height={230} />
    </figure>
  );
}

export default function App() {
  const [task, setTask] = useState("cheetah-run");
  const active = useActiveSection();
  const pick = (t: string) => {
    setTask(t);
    document.getElementById("player")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const v = site.views;

  return (
    <>
      <nav className="topbar" aria-label="Sections">
        <div className="inner">
          <a className="brand" href="#top">Trust Horizon</a>
          {SECTIONS.map((s) => <a key={s.id} href={"#" + s.id}>{s.t}</a>)}
          <a href={REPO}>Code</a>
        </div>
      </nav>

      <div className="frame-grid">
        <aside className="toc" aria-label="Contents">
          <a className="brand" href="#top">Trust Horizon</a>
          <ol>
            {SECTIONS.map((s) => <li key={s.id}><a href={"#" + s.id} aria-current={active === s.id}>{s.t}</a></li>)}
          </ol>
          <div className="links">
            <a href={REPO}>Code and results</a>
            <a href={PAPER}>The paper</a>
          </div>
        </aside>

        <main id="top">
          <header className="hero">
            <div className="hero-top">
              <h1>When do world model dreams break?</h1>
              <div>
                <p className="finding">A world model's own uncertainty signals are good alarms, not good forecasts.</p>
                <p className="byline">Trust Horizon, by Vinod Anbalagan. Final project for Stanford's probability for AI course, November 2026.</p>
                <div className="btn-row">
                  <a href={REPO}>Code and data</a>
                  <a href={PAPER}>Hansen and Wang (2026)</a>
                </div>
              </div>
            </div>

            <section className="summary" aria-label="Summary">
              <div>
                <h3>What we asked</h3>
                <p>A world model imagines the future one frame at a time, and sooner or later the dream goes wrong. Its own uncertainty signals are known to flag which tasks go wrong. We asked whether they can give the probability that a particular dream is about to break, and how long a dream can be trusted.</p>
              </div>
              <div>
                <h3>What we found</h3>
                <p>They are good alarms, not forecasts. They rank risky tasks well (0.84, reproducing the paper) and, within a task, tell a broken dream from a sound one about {Math.round(v.seen.detection.within_task_and_time * 100)}% of the time. But a break usually comes with no warning, and on tasks the model never saw, 86% of dreams break and the signals say nothing useful.</p>
              </div>
              <div>
                <h3>What it means</h3>
                <p>Use the signals to decide where to trust a world model and to cut a drifting dream short. Do not count on them for advance warning, or on new tasks. A real forecast needs a signal that rises before the error does.</p>
              </div>
            </section>

            <ul className="claims">
              <li><span className="num good">0.84</span><span className="txt">The paper's result reproduces: the signals put tasks in nearly the same order, safest to riskiest, as the real error does (rank correlation 0.84; 1 would be perfect).</span></li>
              <li><span className="num">{v.seen.detection.within_task_and_time.toFixed(2)}</span><span className="txt">Within one task, at the same step, the Alarm picks out the broken dream of a broken and a sound pair {Math.round(v.seen.detection.within_task_and_time * 100)}% of the time (50% would be a coin flip). Useful, but modest.</span></li>
              <li><span className="num bad">0</span><span className="txt">Steps of warning before the median break. Shorter windows are not easier, and a more flexible model finds nothing more.</span></li>
              <li><span className="num bad">86%</span><span className="txt">Of dreams break on the 10 tasks the model never trained on; 6 tasks break from step 1, and the Forecast does no better than a coin flip.</span></li>
            </ul>
          </header>

          <Chapter id="dream" title="How a world model dreams">
            <p className="lede prose">A world model is a learned simulator. Shown a few real frames and the actions taken, it imagines the frames that come next, one at a time, each built on the last. Press play and watch one imagine the future.</p>
            <DreamMachine />
            <div className="two-up">
              <p>Robots and agents use such dreams to plan, or to practise without touching the real world. This one is the 350-million-parameter world model of <a href={PAPER}>Hansen and Wang (2026)</a>, trained on 200 tasks: Atari games, simulated robots, mazes and small games. To imagine a frame it starts from random noise and cleans it up in a few steps, like a photograph developing in a tray.</p>
              <p>Because every imagined frame is built on the one before, small errors pile up, and at some point the dream stops matching reality. The paper calls this a <em>hallucination</em> and shows that the world model's own uncertainty signals predict which tasks hallucinate. We asked the probability question: <strong>at any moment of a dream, what is the chance it is about to break, and how long can you trust it?</strong></p>
            </div>
          </Chapter>

          <Chapter id="broken" title="When is a dream broken?">
            <p className="lede prose">To say a dream has broken, we need a fair way to say "too far off". Here is the idea first, then the real thing.</p>
            <BrokenExplainer />
            <p className="prose">A dream works the same way. Each frame is squeezed into a compact code, its position in the world model's map. The gap is the distance between the imagined code and the real one, and the ruler is the distance between two random moments of the same task. A dream is <strong>broken</strong> once its gap stays above half a ruler for 5 steps in a row.</p>
            <ThreeDreams />
          </Chapter>

          <Chapter id="signals" title="How the world model can tell it is unsure">
            <p className="lede prose">While it dreams, the world model can report three numbers about its own guess, without ever seeing the real frames. So a robot could compute them about a future that has not happened yet. Switch between a moment where it is sure and one where it is not.</p>
            <SignalsExplainer />
            <p className="prose">The paper showed that these signals, averaged over a whole dream, tell <em>which tasks</em> the world model dreams badly. We asked a sharper question: at a single moment of a single dream, do they say whether this dream is about to break?</p>
          </Chapter>

          <Chapter id="data" title="The data">
            <DataTable />
          </Chapter>

          <Chapter id="predictions" title="Our two predictions">
            <p className="lede prose">From those numbers, at every step, we compute two chances: the <b>Forecast</b>, that this dream breaks within the next 8 steps, and the <b>Alarm</b>, that it has already broken. Watch them follow the cheetah dream.</p>
            <PredictionExplainer />
          </Chapter>

          <Chapter id="explore" title="Explore every dream">
            <p className="lede prose">Everything so far in one player, for all 30 tasks: the frames, the gap, the Forecast, the Alarm and the signals. Pick a dream and press play.</p>
            <div className="seg" role="group" aria-label="Example dreams" style={{ marginBottom: "0.8rem" }}>
              {PICKS.map((p) => (
                <button key={p.t} aria-pressed={task === p.t} onClick={() => setTask(p.t)}>
                  {pretty(p.t)} <span style={{ opacity: 0.7 }}>({p.why})</span>
                </button>
              ))}
            </div>
            <DreamPlayer task={task} setTask={setTask} />
            <p className="muted" style={{ fontSize: "var(--t-small)", marginTop: "0.7rem" }}>
              Real model output: the paper's 350M-parameter world model, given 8 real frames and the real actions, imagining up to 96 frames at 64 x 64 pixels. Hover a chart to scrub through the dream.
            </p>

          </Chapter>

          <Chapter id="results" title="Alarms, not forecasts">
            <p className="lede prose">How good are the two predictions? First the game that scores them, then the real scores on all 160 test dreams.</p>
            <PairsGame />
            <Split
              text={<>
                <h3 style={{ marginTop: 0 }}>Make the game fair</h3>
                <p>Drawn from anywhere, the pairs are easy: a pong moment against a maze moment, or step 80 against step 5. A prediction can win those just by knowing that pong breaks more than mazes and that late steps break more. A doctor who says "older smokers get sick more" scores well across a whole hospital without reading anyone's chart.</p>
                <p>So we made the pairs more and more alike, until both moments come from the same task at the same step and only the dream itself differs. Press the buttons on the chart to watch the scores move. The Alarm survives the narrowing; the Forecast mostly does not.</p>
              </>}
              fig={<AurocViews />}
            />
            <Split
              text={<>
                <h3 style={{ marginTop: 0 }}>How much notice does a robot get?</h3>
                <p>For every familiar test dream that broke, we counted the steps between the forecast first passing 50% and the break. The median is zero: most breaks arrive with no warning at all.</p>
                <p>The few long "warnings" are not what they seem. Almost all of them began at the very start of the dream, because the model rates some tasks as risky from step 1.</p>
                <p>Below, the calibration and window checks. A skeptic could say we asked badly: the window was too long, the model too simple. Shorter windows are no easier. Gradient-boosted trees, given each signal's last 8 values so they could find any short pattern before a break, did worse than the logistic model (0.52 against 0.61 within task and step) and collapsed on unseen tasks (0.39).</p>
              </>}
              fig={<WarningHist />}
            />
            <div className="two-up">
              <Reliability />
              <WindowSweep />
            </div>
            <h3>The signals do rise before the break, a little</h3>
            <AlignExplainer />
            <Split
              text={<>
                <p>The most direct test lines every breaking dream up at its break and compares it with unbroken dreams of the same task at the same step. On familiar tasks the signals sit 0.3 to 0.6 standard deviations above their controls from about 5 steps before the break, which meets the lead rule we wrote down in advance.</p>
                <p>Then switch the chart to the extra check, which adds the error itself.</p>
                <div className="aside exploratory">The official break is where the error crosses a threshold and stays there. The error does not jump from zero at that moment; it climbs to it, and the signals climb with it. So this is early detection of a slow drift, not foresight of a sudden failure. Real and useful, but detection.</div>
              </>}
              fig={<AlignedAtBreak />}
            />
          </Chapter>

          <Chapter id="horizon" title="How long can you trust a dream?">
            <p className="lede prose">Follow many dreams at once and count, at every step, how many are still sound. That count, as a share, is the trust horizon.</p>
            <SurvivalLanes />
            <Split
              text={<>
                <p>The same count over all 160 test dreams gives the real curves. A survival curve answers the trust-horizon question directly: of all dreams that start, what share is still sound after <em>t</em> steps?</p>
                <p>Dreams that never broke within their horizon still count, up to the last step we saw. That is what a Kaplan-Meier curve handles: it uses every dream for as long as it was watched.</p>
                <p>On familiar tasks half the dreams are still sound at step {site.survival.seen.pooled.median}, though tasks differ enormously: robot-arm dreams break within a handful of steps, some maze dreams never do. On unseen tasks half are broken at step 1.</p>
              </>}
              fig={<SurvivalCurves />}
            />
          </Chapter>

          <Chapter id="unseen" title="Tasks the model never saw">
            <div className="two-up" style={{ marginTop: 0 }}>
              <p className="lede" style={{ margin: 0 }}>The 10 held-out tasks were dreamed early but sealed: only file fingerprints were kept, the analysis code refused to open them, and the final test opened them once.</p>
              <p style={{ margin: 0 }}>86% of dreams break, and on 6 of the 10 tasks the typical dream is broken from step 1, though several are small variations of familiar tasks. The model cannot even reproduce the first frame of a scene it has not seen. The paper calls this perceptual hallucination: the part of the world model that compresses frames rebuilds familiar scenes almost perfectly but unseen scenes poorly (38.3 against 17.3 on its image-quality score, where higher is better). With most dreams broken before they start, the forecast falls to {v.unseen.forecast.beyond_time.toFixed(2)}.</p>
            </div>
            <UnseenGallery onPick={pick} />
          </Chapter>

          <Chapter id="distance" title="Distance from the training data">
            <Split
              text={<>
                <p className="lede">A world model should be most trustworthy near what it was trained on. Methods such as Mind Dreamer (ICML 2026) dream from novel states on purpose, so the price of novelty matters.</p>
                <p>For 5 familiar tasks, MMBench2 also has episodes recorded after the model was trained, by a random policy, a do-nothing policy and an expert. For each new starting moment we measured how far it is from the most similar moments the world model trained on (the average distance to its 5 closest matches among 2,000 training clips), then dreamed 96 steps from it.</p>
                <p>This does not show that distance is harmless. A random policy in these simple games visits the same kinds of screens the training data covered, so the test had almost no range. The genuinely far case is the unseen tasks, where distance clearly matters. Within a familiar task it was not shown.</p>
              </>}
              fig={<DistanceSurvival />}
            />
          </Chapter>

          <Chapter id="meaning" title="What it means">
            <div className="two-up" style={{ marginTop: 0 }}>
              <div>
                <p className="lede">The signals work like a fuel gauge, not a smoke alarm. They show which trips are risky and how far a dream has already drifted, but they give no advance notice of a sudden failure, and on unfamiliar ground the gauge reads nothing useful.</p>
                <p>A real forecast needs a signal that rises <em>before</em> the error does. Candidates worth testing: disagreement between independently trained models, uncertainty over longer stretches of the dream, or a predictor trained on the error itself.</p>
              </div>
              <div>
                <p>If you plan or train inside a world model's dreams:</p>
                <ul className="tight">
                  <li>use the signals to choose <em>where</em> to trust the model: which tasks, which kinds of scenes;</li>
                  <li>use them as a running gauge of drift and cut dreams short when they climb, knowing you will learn of a break at about the moment it happens;</li>
                  <li>do not expect them to warn you on new tasks; there the first frame is already wrong, and a separate novelty check is needed before dreaming at all.</li>
                </ul>
              </div>
            </div>
          </Chapter>

          <Chapter id="probability" title="The probability behind it">
            <p className="lede prose">Every number on this page comes from a handful of ideas from the Probability for AI course. Here they are, in the form this project uses them.</p>
            <ProbabilityCards />
          </Chapter>

          <Chapter id="honest" title="How we kept ourselves honest">
            <div className="prose"><h3>Reproducing the paper first</h3>
                <p>Our first reproduction failed: Spearman 0.26 to 0.42 against the paper's 0.8. An audit against the authors' evaluation code found four settings copied from their interactive demo instead (float16 instead of bfloat16, 8 denoising steps instead of 4, and two signals computed from the wrong number of seeds). With those fixed and committed before the rerun, the task-level correlation came out at <strong>0.84 for u<sub>r</sub> and 0.85 for u<sub>f</sub></strong>. Everything below rests on that matched setup.</p></div>
            <p className="lede prose">Three labels, several smoothing choices and many ways to pool make it easy to try options until one looks good. Four habits guarded against that.</p>
            <div className="cards">
              <div className="card"><h4>Guesses before data</h4><p>Six predictions with confidences were committed before any dream existed and never edited.</p></div>
              <div className="card"><h4>A protocol written ahead</h4><p>Every measurement decision was committed before the data it governs. The few made after seeing related results are marked exploratory.</p></div>
              <div className="card"><h4>A sealed test set</h4><p>The unseen tasks were opened once, after a hash check, by code that then refused to run again.</p></div>
              <div className="card"><h4>Known-answer tests</h4><p>Every analysis first ran on fake dreams with a planted answer. They caught fake warning from a smoothing choice and an importance measure that picked the wrong signal.</p></div>
            </div>
            <div className="two-tables">
              <div>
                <h3 style={{ marginTop: 0 }}>How the guesses turned out</h3>
                <div className="tablewrap">
                  <table>
                    <thead><tr><th>Guess (confidence)</th><th>Result</th><th></th></tr></thead>
                    <tbody>
                      <tr><td>Using all signals together beats the best single signal at getting the chances right (70%)</td><td>0.079 against 0.092</td><td className="verdict-yes">yes</td></tr>
                      <tr><td>Calibration within 0.1 in every well-filled bin (55%)</td><td>gap 0.27</td><td className="verdict-no">no</td></tr>
                      <tr><td>Median warning of at least 3 steps (40%)</td><td>0 steps</td><td className="verdict-no">no</td></tr>
                      <tr><td>Time to failure falls steadily with distance (60%)</td><td>not monotone</td><td className="verdict-no">no</td></tr>
                      <tr><td>Unseen forecast within 0.05 of familiar (40%)</td><td>0.66 against 0.84</td><td className="verdict-no">no</td></tr>
                      <tr><td>u<sub>s</sub> matters most, by ablation (50%)</td><td>removing it costs 0.069 AUROC</td><td className="verdict-yes">yes</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
              <div>
                <h3 style={{ marginTop: 0 }}>Mistakes we caught</h3>
                <div className="tablewrap">
                  <table>
                    <thead><tr><th>Mistake</th><th>How it showed up</th><th>Fix</th></tr></thead>
                    <tbody>
                      <tr><td>Evaluation settings copied from the authors' demo</td><td>Reproduction at 0.26 to 0.42 instead of 0.8</td><td>Audit against the paper's code, rerun</td></tr>
                      <tr><td>The first checkpoint had been finetuned on the "unseen" tasks too</td><td>Reading its README before the main run</td><td>Base checkpoint throughout</td></tr>
                      <tr><td>A trailing median noticed breaks 2 steps late</td><td>Fake warning in a known-answer test</td><td>Centered median</td></tr>
                      <tr><td>Still scenes gave an infinite frozen-frame score</td><td>3 of 20 tasks in the first run</td><td>A small noise floor</td></tr>
                      <tr><td>The paper's label mostly tracked scene motion</td><td>Negative correlations within tasks</td><td>Direct-error label, marked exploratory</td></tr>
                      <tr><td>"Largest weight" picked the wrong signal</td><td>Known-answer test</td><td>Importance by ablation</td></tr>
                      <tr><td>The Alarm's 0.84 was pooled across tasks</td><td>The within-task view</td><td>Every AUROC reported three ways</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <p className="prose">Every quick, unplanned number was wrong at least once; numbers from the tested pipeline were not.</p>
          </Chapter>

          <footer className="foot">
            <div>
              <h4>Limits</h4>
              <p>One model (the 350M base checkpoint), one benchmark, and whole-image error measures, which underweight small objects such as a maze agent wandering off its path. Familiar tasks have 8 test dreams each, so several numbers rest on few dreams. The direct-error label and two checks were added after seeing earlier results and are labelled exploratory. Each dream follows one noise seed.</p>
            </div>
            <div>
              <h4>Code and data</h4>
              <p>Pipeline, protocol, pre-registration and every result file: <a href={REPO}>github.com/VinodAnbalagan/trust-horizon</a>. Built on Hansen and Wang, <a href={PAPER}>Hallucination in World Models is Predictable and Preventable</a> (2026), with their <a href={PAPER_PAGE}>project page</a>, <a href={PAPER_CODE}>code</a>, checkpoints and MMBench2 data. This page bundles its data and makes no network requests.</p>
              <h4>Cost</h4>
              <p>All GPU work cost about $4.03 over three days on rented cards. Every analysis after the dreams ran free on a laptop CPU.</p>
            </div>
            <div>
              <h4>Use of AI tools</h4>
              <p>Claude (Anthropic) was used throughout: to read the paper and its code, write and test the analysis pipeline, run and check the analyses, write the explanatory document, and build this site. The pre-registered guesses are the author's own, written before any data. The research questions and the decisions recorded in the protocol were worked out in conversation with Claude and decided by the author. Every number on this page is computed by the code in the repository.</p>
            </div>
          </footer>
        </main>
      </div>
    </>
  );
}
