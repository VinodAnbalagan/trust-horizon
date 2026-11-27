import { useEffect, useState } from "react";
import DreamPlayer from "./components/DreamPlayer";
import LineChart from "./components/LineChart";
import {
  AlignedAtBreak, AurocViews, DistanceSurvival, Filmstrip, Reliability, SurvivalCurves, UnseenGallery, WarningHist, WindowSweep,
} from "./components/Figures";
import { dreams, pretty, site } from "./data";
import { ProbabilityCards } from "./components/Probability";

const REPO = "https://github.com/VinodAnbalagan/trust-horizon";
const PAPER = "https://arxiv.org/abs/2606.27326";
const PAPER_PAGE = "https://www.nicklashansen.com/mmbench2";
const PAPER_CODE = "https://github.com/nicklashansen/mmbench2";

const SECTIONS = [
  { id: "dream", t: "How a world model dreams" },
  { id: "signals", t: "The three signals" },
  { id: "broken", t: "What counts as broken" },
  { id: "probability", t: "The probability behind it" },
  { id: "results", t: "Alarms, not forecasts" },
  { id: "horizon", t: "The trust horizon" },
  { id: "unseen", t: "Unseen tasks" },
  { id: "distance", t: "Distance from training" },
  { id: "meaning", t: "What it means" },
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

            <ul className="claims">
              <li><span className="num good">0.84</span><span className="txt">The paper's result reproduces: the signals rank tasks by how badly the model dreams them (Spearman 0.84 and 0.85).</span></li>
              <li><span className="num">{v.seen.detection.within_task_and_time.toFixed(2)}</span><span className="txt">Within one task at the same step, they tell a broken dream from a sound one with AUROC {v.seen.detection.within_task_and_time.toFixed(2)}. Useful, but modest.</span></li>
              <li><span className="num bad">0</span><span className="txt">Steps of warning before the median break. Shorter windows are not easier, and a more flexible model finds nothing more.</span></li>
              <li><span className="num bad">86%</span><span className="txt">Of dreams break on the 10 tasks the model never trained on; 6 tasks break from step 1, and the forecast falls to chance ({v.unseen.forecast.beyond_time.toFixed(2)}).</span></li>
            </ul>
          </header>

          <Chapter id="dream" title="How a world model dreams">
            <Split
              text={<>
                <p className="lede">A world model is a learned simulator. Shown a few real frames and a list of actions, it predicts the frames that follow. When it keeps feeding its own predictions back in, it is dreaming.</p>
                <p>Robots and agents use such dreams to plan, or to practise without touching the real world. <a href={PAPER}>Hansen and Wang (2026)</a> trained a 350-million-parameter world model in the style of Dreamer 4 on MMBench2: 200 tasks across 10 domains, from Atari and DeepMind Control to robot arms, mazes and small pygame games, with 10 more tasks held out.</p>
                <p>A tokenizer squeezes each 64 x 64 frame into a compact code, a <em>latent</em>, the way a zip file holds a picture. A dynamics Transformer predicts the next latent from the past ones and the action. It starts from random noise and cleans it up in a few steps, like a photograph developing in a tray. That is flow matching.</p>
                <p>Every predicted frame is a little wrong, and the next one is built on it. At some point the dream stops resembling what happened: the agent is somewhere else, the ball has vanished, the scene has frozen. The paper calls that drift a <em>hallucination</em> and shows that the model's own signals predict which tasks will hallucinate.</p>
                <p>We asked the probability question on top of theirs: <strong>at any step of a dream, what is the chance it breaks within the next few steps, and how long can you trust it?</strong> We call that span the trust horizon.</p>
                <div className="flow" aria-label="How one dream step is made">
                  <span className="node">8 real frames</span><span className="arrow">→</span>
                  <span className="node">tokenizer</span><span className="arrow">→</span>
                  <span className="node">latents</span><span className="arrow">→</span>
                  <span className="node">dynamics + action</span><span className="arrow">→</span>
                  <span className="node dream">next latent</span><span className="arrow">→</span>
                  <span className="node">decoder</span><span className="arrow">→</span>
                  <span className="node dream">dreamed frame</span><span className="arrow">↺ fed back in</span>
                </div>
              </>}
              fig={<Filmstrip />}
            />
          </Chapter>

          <Chapter id="signals" title="Three ways to ask the model whether it is sure">
            <p className="lede prose">Each dream step yields three numbers that need no ground truth, so a robot could compute them about a future it has not seen.</p>
            <div className="cards">
              <div className="card ur">
                <h4>u<sub>r</sub>, round trip</h4>
                <p>Decode the predicted latent to an image and encode it again. How far did it move?</p>
                <p className="everyday">Translate a sentence to French and back. A well-formed sentence survives unchanged.</p>
              </div>
              <div className="card uf">
                <h4>u<sub>f</sub>, flow instability</h4>
                <p>While cleaning up the noise, did the model keep changing its guess of the clean frame?</p>
                <p className="everyday">A student who keeps rewriting an answer is less sure than one who writes it once.</p>
              </div>
              <div className="card us">
                <h4>u<sub>s</sub>, seed spread</h4>
                <p>Do two different random starts lead to the same next frame?</p>
                <p className="everyday">Ask two witnesses. If their stories differ, trust neither.</p>
              </div>
            </div>
            <div className="two-up" style={{ marginTop: 0 }}>
              <p>A busy scene makes every signal larger even when the dream is fine, so, as in the paper, each signal is divided by how much the task's real latents normally move per step. A value of 2 means "twice as unsure as this scene usually moves".</p>
              <p>The paper averaged the signals over a 16-step dream and compared tasks. That answers <em>which tasks</em> to worry about. We wanted the per-step, per-dream version: do the numbers at step <em>t</em> say anything about this dream's next few steps?</p>
            </div>
          </Chapter>

          <Chapter id="broken" title="What counts as broken">
            <Split
              text={<>
                <p className="lede">To forecast a break you must first define one, and the definition turned out to matter more than any model.</p>
                <p>Our primary label uses a natural yardstick, the <em>chance error</em>: the typical distance between two unrelated moments of the same task. If you are guessing where a friend is in a city, the chance error is the distance between two random spots in that city. Being off by half of that is a bad guess. A dream is broken once its error stays at or above half the chance error.</p>
                <p>We smooth the error with a 5-step median and call the break at the first of 5 broken steps in a row, so one wild frame cannot trigger it, and a dream that touches the line and recovers has not broken.</p>
                <p>The paper's own label, "worse than repeating the last real frame", turned out to rise and fall with how much the scene moves, so it partly measures motion rather than dream quality. Under it almost nothing worked.</p>
                <h3>Reproducing the paper first</h3>
                <p>Our first reproduction failed: Spearman 0.26 to 0.42 against the paper's 0.8. An audit against the authors' evaluation code found four settings copied from their interactive demo instead (float16 instead of bfloat16, 8 denoising steps instead of 4, and two signals computed from the wrong number of seeds). With those fixed and committed before the rerun, the task-level correlation came out at <strong>0.84 for u<sub>r</sub> and 0.85 for u<sub>f</sub></strong>. Everything below rests on that matched setup.</p>
              </>}
              fig={<BreakDefinition />}
            />
          </Chapter>

          <Chapter id="probability" title="The probability behind it">
            <p className="lede prose">Every number on this page comes from a handful of ideas from the Probability for AI course. Here they are, in the form this project uses them.</p>
            <ProbabilityCards />
          </Chapter>

          <Chapter id="results" title="Alarms, not forecasts">
            <Split
              text={<>
                <p className="lede">We trained a small, readable probability model on 320 dreams and tested it on 160 others. It sees 11 numbers per step, all known at that step, and gives two probabilities: has the dream already broken, and will it break within the next 8 steps?</p>
                <p>Pooled over all test steps, both look strong. But a pooled score rewards knowing that pong breaks more than mazes, and that step 80 is riskier than step 5. A doctor who says "older smokers get sick more" scores well across a whole hospital without reading anyone's chart.</p>
                <p>So we narrowed the comparison until only the dream in front of the model was left. Press the buttons on the chart to watch the scores move. Detection survives the narrowing; the forecast mostly does not.</p>
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
            <Split
              text={<>
                <h3 style={{ marginTop: 0 }}>The signals do rise before the break, a little</h3>
                <p>The most direct test lines every breaking dream up at its break and compares it with unbroken dreams of the same task at the same step. On familiar tasks the signals sit 0.3 to 0.6 standard deviations above their controls from about 5 steps before the break, which meets the lead rule we wrote down in advance.</p>
                <p>Then switch the chart to the extra check, which adds the error itself.</p>
                <div className="aside exploratory">The official break is where the error crosses a threshold and stays there. The error does not jump from zero at that moment; it climbs to it, and the signals climb with it. So this is early detection of a slow drift, not foresight of a sudden failure. Real and useful, but detection.</div>
              </>}
              fig={<AlignedAtBreak />}
            />
          </Chapter>

          <Chapter id="horizon" title="How long can you trust a dream?">
            <Split
              text={<>
                <p className="lede">A survival curve answers the trust-horizon question directly: of all dreams that start, what share is still sound after <em>t</em> steps?</p>
                <p>Dreams that never broke within their horizon still count, up to the last step we saw. That is what a Kaplan-Meier curve handles: it uses every dream for as long as it was watched.</p>
                <p>On familiar tasks half the dreams are still sound at step {site.survival.seen.pooled.median}, though tasks differ enormously: robot-arm dreams break within a handful of steps, some maze dreams never do. On unseen tasks half are broken at step 1.</p>
              </>}
              fig={<SurvivalCurves />}
            />
          </Chapter>

          <Chapter id="unseen" title="Tasks the model never saw">
            <div className="two-up" style={{ marginTop: 0 }}>
              <p className="lede" style={{ margin: 0 }}>The 10 held-out tasks were dreamed early but sealed: only file fingerprints were kept, the analysis code refused to open them, and the final test opened them once.</p>
              <p style={{ margin: 0 }}>86% of dreams break, and on 6 of the 10 tasks the typical dream is broken from step 1, though several are small variations of familiar tasks. The model cannot even reproduce the first frame of a scene it has not seen. The paper calls this perceptual hallucination, with the tokenizer's reconstruction quality falling from 38.3 dB on seen tasks to 17.3 dB on unseen ones. With most dreams broken before they start, the forecast falls to {v.unseen.forecast.beyond_time.toFixed(2)}.</p>
            </div>
            <UnseenGallery onPick={pick} />
          </Chapter>

          <Chapter id="distance" title="Distance from the training data">
            <Split
              text={<>
                <p className="lede">A world model should be most trustworthy near what it was trained on. Methods such as Mind Dreamer (ICML 2026) dream from novel states on purpose, so the price of novelty matters.</p>
                <p>For 5 familiar tasks, MMBench2 also has episodes recorded after the model was trained, by a random policy, a do-nothing policy and an expert. We measured each start state's distance to 2,000 training clips (mean distance to its 5 nearest neighbours) and dreamed 96 steps from each.</p>
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

          <Chapter id="honest" title="How we kept ourselves honest">
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
                      <tr><td>The combined model beats the best single signal on Brier score (70%)</td><td>0.079 against 0.092</td><td className="verdict-yes">yes</td></tr>
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
                      <tr><td>Detection 0.84 was pooled across tasks</td><td>The within-task view</td><td>Every AUROC reported three ways</td></tr>
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
