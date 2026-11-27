import { useState } from "react";
import DreamPlayer from "./components/DreamPlayer";
import { AlignedAtBreak, AurocViews, DistanceSurvival, Reliability, SurvivalCurves, UnseenGallery, WarningHist, WindowSweep } from "./components/Figures";
import { pretty, site } from "./data";

const REPO = "https://github.com/VinodAnbalagan/trust-horizon";
const PAPER = "https://arxiv.org/abs/2606.27326";
const PAPER_PAGE = "https://www.nicklashansen.com/mmbench2";
const PAPER_CODE = "https://github.com/nicklashansen/mmbench2";

const PICKS: { t: string; why: string }[] = [
  { t: "cheetah-run", why: "slow drift" },
  { t: "pygame-pong", why: "breaks at step 4" },
  { t: "pygame-point-maze-var1", why: "never breaks" },
  { t: "cup-catch-var1", why: "unseen, broken at step 1" },
  { t: "pygame-foraging", why: "unseen, dissolves" },
];

function Chapter({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section className="chapter" id={id} aria-labelledby={id + "-h"}>
      <div className="col"><h2 id={id + "-h"}>{title}</h2></div>
      {children}
    </section>
  );
}

export default function App() {
  const [task, setTask] = useState("cheetah-run");
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
          <a href="#dream">The dream</a>
          <a href="#signals">Signals</a>
          <a href="#results">Alarms, not forecasts</a>
          <a href="#horizon">Trust horizon</a>
          <a href="#unseen">Unseen tasks</a>
          <a href="#distance">Distance</a>
          <a href="#honest">How we checked</a>
          <a href={REPO}>Code</a>
        </div>
      </nav>

      <main className="page" id="top">
        <header className="hero wide">
          <h1>When do world model dreams break?</h1>
          <p className="byline">Trust Horizon, by Vinod Anbalagan. Final project for Stanford's probability for AI course, November 2026.</p>
          <p className="finding">A world model's own uncertainty signals are good alarms, not good forecasts.</p>

          <div className="seg" role="group" aria-label="Example dreams" style={{ marginBottom: "1rem" }}>
            {PICKS.map((p) => (
              <button key={p.t} aria-pressed={task === p.t} onClick={() => setTask(p.t)}>
                {pretty(p.t)} <span style={{ opacity: 0.7 }}>({p.why})</span>
              </button>
            ))}
          </div>
          <DreamPlayer task={task} setTask={setTask} />
          <p className="muted" style={{ fontFamily: "var(--sans)", fontSize: "var(--t-small)", marginTop: "0.9rem", maxWidth: "var(--measure)" }}>
            Each dream is real model output: the paper's 350M-parameter world model, given 8 real frames and the real actions, imagining up to 96 frames at 64 x 64 pixels. Hover a chart to scrub through the dream.
          </p>
        </header>

        <div className="col">
          <ul className="claims">
            <li><span className="num good">0.84</span><span className="txt">The paper's result reproduces. The signals rank tasks by how badly the model dreams them (Spearman 0.84 and 0.85, against the paper's about 0.8).</span></li>
            <li><span className="num">{v.seen.detection.within_task_and_time.toFixed(2)}</span><span className="txt">Within one task, at the same step, they tell a broken dream from a sound one with AUROC {v.seen.detection.within_task_and_time.toFixed(2)}. Useful, but modest: most of their skill is telling risky tasks apart.</span></li>
            <li><span className="num bad">0</span><span className="txt">Steps of warning before the median break. Shorter forecast windows are not easier, and a more flexible model finds nothing more.</span></li>
            <li><span className="num bad">86%</span><span className="txt">Of dreams break on the 10 tasks the model never trained on; 6 of those tasks break from the very first step, and the forecast falls to chance ({v.unseen.forecast.beyond_time.toFixed(2)}).</span></li>
          </ul>
        </div>

        <Chapter id="dream" title="How a world model dreams">
          <div className="col">
            <p className="lede">A world model is a learned simulator. Shown a few real frames and a list of actions, it predicts the frames that follow. When it keeps feeding its own predictions back in, it is dreaming.</p>
            <p>
              Robots and agents use such dreams to plan, or to practise without touching the real world. <a href={PAPER}>Hansen and Wang (2026)</a> trained a 350-million-parameter world model in the style of Dreamer 4 on MMBench2: 200 tasks across 10 domains, from Atari and DeepMind Control to robot arms, mazes and small pygame games, with 10 more tasks held out. A tokenizer squeezes each 64 x 64 frame into a compact code, a <em>latent</em>, the way a zip file holds a picture. A dynamics Transformer predicts the next latent from the past ones and the action.
            </p>
            <p>
              It does not output the next latent in one shot. It starts from random noise and cleans it up in a few steps, like a photograph developing in a tray. That is flow matching. The model is given 8 real frames, predicts frame 9, appends it, predicts frame 10, and so on. Every predicted frame is a little wrong, and the next one is built on it: a photocopy of a photocopy. At some point the dream stops resembling what happened. The agent is somewhere else, the ball has vanished, the scene has frozen.
            </p>
            <p>
              The paper calls that drift a <em>hallucination</em> and shows that the model's own signals predict which tasks will hallucinate. We asked the probability question on top of theirs: <strong>at any step of a dream, what is the chance it breaks within the next few steps, and how long can you trust it?</strong> We call that span the trust horizon.
            </p>
          </div>
        </Chapter>

        <Chapter id="signals" title="Three ways to ask the model whether it is sure">
          <div className="col">
            <p className="lede">Each dream step yields three numbers that need no ground truth, so a robot could compute them about a future it has not seen.</p>
            <div className="tablewrap">
              <table>
                <thead><tr><th>Signal</th><th>Question it asks</th><th>Everyday version</th></tr></thead>
                <tbody>
                  <tr><td><b style={{ color: "var(--s-ur)" }}>u<sub>r</sub></b> round trip</td><td>Decode the predicted latent to an image and encode it again. How far did it move?</td><td>Translate a sentence to French and back. A well-formed sentence survives unchanged.</td></tr>
                  <tr><td><b style={{ color: "var(--s-uf)" }}>u<sub>f</sub></b> flow instability</td><td>While cleaning up the noise, did the model keep changing its guess?</td><td>A student who keeps rewriting an answer is less sure than one who writes it once.</td></tr>
                  <tr><td><b style={{ color: "var(--s-us)" }}>u<sub>s</sub></b> seed spread</td><td>Do two different random starts lead to the same next frame?</td><td>Ask two witnesses. If their stories differ, trust neither.</td></tr>
                </tbody>
              </table>
            </div>
            <p>
              A busy scene makes every signal larger even when the dream is fine, so, as in the paper, each signal is divided by how much the task's real latents normally move per step. A value of 2 means "twice as unsure as this scene usually moves".
            </p>
            <p>
              The paper averaged the signals over a 16-step dream and compared tasks. That answers <em>which tasks</em> to worry about. We wanted the per-step, per-dream version: do the numbers at step <em>t</em> say anything about this dream's next few steps?
            </p>

            <h3 id="broken">What counts as broken</h3>
            <p>
              To forecast a break you must first define one, and the definition turned out to matter more than any model. Our primary label compares the dream with a natural yardstick: the <em>chance error</em>, the typical distance between two unrelated moments of the same task. If you are guessing where a friend is in a city, the chance error is the distance between two random spots in that city. Being off by half of that is a bad guess. A dream is broken once its error stays at or above half the chance error.
            </p>
            <p>
              We smooth the error with a 5-step median and call the break at the first of 5 broken steps in a row, so one wild frame cannot trigger it and a dream that touches the line and recovers has not broken. The paper's own label, "worse than repeating the last real frame", turned out to rise and fall with how much the scene moves, so it partly measures motion rather than dream quality. We report it too; under it almost nothing worked.
            </p>

            <h3>Reproducing the paper first</h3>
            <p>
              Our first reproduction failed: Spearman 0.26 to 0.42 against the paper's 0.8. A line-by-line audit against the authors' evaluation code found four settings we had copied from their interactive demo instead (float16 instead of bfloat16, 8 denoising steps instead of 4, and two ways of computing the signals from the wrong number of seeds). With those fixed, recorded and committed before the rerun, the task-level correlation came out at <strong>0.84 for u<sub>r</sub> and 0.85 for u<sub>f</sub></strong>. Everything below rests on that matched setup.
            </p>
          </div>
        </Chapter>

        <Chapter id="results" title="Alarms, not forecasts">
          <div className="col">
            <p className="lede">We trained a small, readable probability model on 320 dreams and tested it on 160 others. It sees 11 numbers per step, all known at that step, and gives two probabilities: has the dream already broken, and will it break within the next 8 steps?</p>
            <p>
              Pooled over all test steps, both look strong. But a pooled score rewards knowing that pong breaks more than mazes, and that step 80 is riskier than step 5. A doctor who says "older smokers get sick more" scores well across a whole hospital without reading anyone's chart. So we kept narrowing the comparison until only the dream in front of the model was left.
            </p>
          </div>
          <div className="col"><AurocViews /></div>
          <div className="col">
            <p>
              Detection survives the narrowing; the forecast mostly does not. Then the question that matters for a robot: how much notice does it get?
            </p>
          </div>
          <div className="col"><WarningHist /></div>
          <div className="wide two-up">
            <Reliability />
            <WindowSweep />
          </div>
          <div className="col">
            <p style={{ marginTop: "2rem" }}>
              A skeptic could say we asked badly: the window was too long, the model too simple. Shorter windows are no easier, as the window sweep above shows. Gradient-boosted trees, given each signal's last 8 values so they could find any short pattern before a break, did worse than the logistic model (0.52 against 0.61 within task and step) and collapsed on unseen tasks (0.39).
            </p>
            <h3>The signals do rise before the break, a little</h3>
            <p>
              The most direct test lines every breaking dream up at its break and compares it with unbroken dreams of the same task at the same step. On familiar tasks the signals sit 0.3 to 0.6 standard deviations above their controls from about 5 steps before the break, which meets the lead rule we wrote down in advance.
            </p>
          </div>
          <div className="col"><AlignedAtBreak /></div>
          <div className="col">
            <div className="aside exploratory">
              The official break is where the error crosses a threshold and stays there. The error does not jump from zero at that moment; it climbs to it, and the signals climb with it. So this is early detection of a slow drift, not foresight of a sudden failure. Real and useful, but detection.
            </div>
          </div>
        </Chapter>

        <Chapter id="horizon" title="How long can you trust a dream?">
          <div className="col">
            <p className="lede">A survival curve answers the trust-horizon question directly: of all dreams that start, what share is still sound after <em>t</em> steps?</p>
            <p>Dreams that never broke within their horizon still count, up to the last step we saw. That is what a Kaplan-Meier curve handles: it uses every dream for as long as it was watched.</p>
          </div>
          <div className="col"><SurvivalCurves /></div>
        </Chapter>

        <Chapter id="unseen" title="Tasks the model never saw">
          <div className="col">
            <p className="lede">
              The 10 held-out tasks were dreamed early but sealed: only file fingerprints were kept, the analysis code refused to open them, and the final test opened them once.
            </p>
            <p>
              The result is stark. 86% of dreams break, and on 6 of the 10 tasks the typical dream is broken from step 1. Several are small variations of familiar tasks, a new cup or a new maze, which makes it more striking. The model cannot even reproduce the first frame of a scene it has not seen. The paper calls this perceptual hallucination and reports the tokenizer's reconstruction quality falling from 38.3 dB on seen tasks to 17.3 dB on unseen ones. There is nothing left to forecast when most dreams are broken before they start: the forecast falls to {v.unseen.forecast.beyond_time.toFixed(2)}.
            </p>
          </div>
          <UnseenGallery onPick={pick} />
        </Chapter>

        <Chapter id="distance" title="Distance from the training data">
          <div className="col">
            <p className="lede">A world model should be most trustworthy near what it was trained on. Methods such as Mind Dreamer (ICML 2026) dream from novel states on purpose, so the price of novelty matters.</p>
            <p>
              For 5 familiar tasks, MMBench2 also has episodes recorded after the model was trained, by a random policy, a do-nothing policy and an expert. We measured each start state's distance to 2,000 training clips (mean distance to its 5 nearest neighbours) and dreamed 96 steps from each.
            </p>
          </div>
          <div className="col"><DistanceSurvival /></div>
          <div className="col">
            <p>
              This does not show that distance is harmless. A random policy in these simple games visits the same kinds of screens the training data covered, so the test had almost no range. The genuinely far case is the unseen tasks above, where distance clearly matters. Within a familiar task it was not shown.
            </p>
          </div>
        </Chapter>

        <Chapter id="meaning" title="What it means">
          <div className="col">
            <p className="lede">The signals work like a fuel gauge, not a smoke alarm. They show which trips are risky and how far a dream has already drifted, but they give no advance notice of a sudden failure, and on unfamiliar ground the gauge reads nothing useful.</p>
            <p>If you plan or train inside a world model's dreams:</p>
            <ul>
              <li>use the signals to choose <em>where</em> to trust the model: which tasks, which kinds of scenes;</li>
              <li>use them as a running gauge of drift and cut dreams short when they climb, knowing you will learn of a break at about the moment it happens;</li>
              <li>do not expect them to warn you on new tasks; there the first frame is already wrong, and a separate novelty check is needed before dreaming at all.</li>
            </ul>
            <p>
              A real forecast needs a signal that rises <em>before</em> the error does. Candidates worth testing: disagreement between independently trained models, uncertainty over longer stretches of the dream, or a predictor trained on the error itself.
            </p>
          </div>
        </Chapter>

        <Chapter id="honest" title="How we kept ourselves honest">
          <div className="col">
            <p className="lede">Three labels, several smoothing choices and many ways to pool make it easy to try options until one looks good. Four habits guarded against that.</p>
            <ul>
              <li><strong>Guesses before data.</strong> Six predictions with confidences were committed before any dream existed and never edited.</li>
              <li><strong>A protocol written ahead.</strong> Every measurement decision was committed before the data it governs. The few made after seeing related results are marked exploratory, here and in the repository.</li>
              <li><strong>A sealed test set.</strong> The unseen tasks were opened once, after a hash check, by code that then refused to run again.</li>
              <li><strong>Known-answer tests.</strong> Every analysis first ran on fake dreams with a planted answer. They caught a smoothing choice that produced 2 steps of fake warning and an importance measure that picked the wrong signal.</li>
            </ul>
            <h3>How the guesses turned out</h3>
            <div className="tablewrap">
              <table>
                <thead><tr><th>Guess, with the confidence written down</th><th>Result</th><th>Outcome</th></tr></thead>
                <tbody>
                  <tr><td>The combined model beats the best single signal on Brier score (70%)</td><td>0.079 against 0.092</td><td className="verdict-yes">yes</td></tr>
                  <tr><td>Calibration within 0.1 in every well-filled bin (55%)</td><td>gap 0.27</td><td className="verdict-no">no</td></tr>
                  <tr><td>Median warning of at least 3 steps (40%)</td><td>0 steps</td><td className="verdict-no">no</td></tr>
                  <tr><td>Time to failure falls steadily with distance (60%)</td><td>not monotone</td><td className="verdict-no">no</td></tr>
                  <tr><td>Unseen forecast within 0.05 of familiar (40%)</td><td>0.66 against 0.84</td><td className="verdict-no">no</td></tr>
                  <tr><td>u<sub>s</sub> matters most (50%), by ablation</td><td>removing u<sub>s</sub> costs 0.069 AUROC</td><td className="verdict-yes">yes</td></tr>
                </tbody>
              </table>
            </div>
            <h3>Mistakes we caught</h3>
            <div className="tablewrap">
              <table>
                <thead><tr><th>Mistake</th><th>How it showed up</th><th>Fix</th></tr></thead>
                <tbody>
                  <tr><td>Evaluation settings copied from the authors' demo</td><td>Reproduction at 0.26 to 0.42 instead of 0.8</td><td>Audit against the paper's code, rerun</td></tr>
                  <tr><td>The first checkpoint had been finetuned on the "unseen" tasks too</td><td>Reading its README before the main run</td><td>Used the base checkpoint throughout</td></tr>
                  <tr><td>A trailing median noticed breaks 2 steps late</td><td>2 steps of fake warning in a known-answer test</td><td>Centered median</td></tr>
                  <tr><td>Still scenes gave an infinite frozen-frame score</td><td>3 of 20 tasks in the first run</td><td>A small noise floor</td></tr>
                  <tr><td>The paper's label mostly tracked scene motion</td><td>Negative correlations within tasks</td><td>Direct-error label, marked exploratory</td></tr>
                  <tr><td>"Largest weight" picked the wrong signal</td><td>Known-answer test</td><td>Importance by ablation</td></tr>
                  <tr><td>Detection 0.84 was pooled across tasks</td><td>The within-task view in the unseen-task analysis</td><td>Every AUROC reported three ways: 0.72 within task and step</td></tr>
                </tbody>
              </table>
            </div>
            <p>Every quick, unplanned number was wrong at least once; numbers from the tested pipeline were not.</p>
          </div>
        </Chapter>

        <footer className="foot">
          <div className="col">
            <h4>Limits</h4>
            <p>One model (the 350M base checkpoint), one benchmark, and whole-image error measures, which underweight small objects such as a maze agent wandering off its path. Familiar tasks have 8 test dreams each, so several numbers rest on few dreams. The direct-error label and two checks were added after seeing earlier results and are labelled exploratory. Each dream follows one noise seed.</p>
            <h4>Cost</h4>
            <p>All GPU work cost about $4.03 over three days on rented RTX 3090, 4090 and RTX PRO 4500 cards. Every analysis after the dreams ran free on a laptop CPU.</p>
            <h4>Code and data</h4>
            <p>
              Pipeline, protocol, pre-registration and every result file: <a href={REPO}>github.com/VinodAnbalagan/trust-horizon</a>. Built on Hansen and Wang, <a href={PAPER}>Hallucination in World Models is Predictable and Preventable</a> (2026), with their <a href={PAPER_PAGE}>project page</a>, <a href={PAPER_CODE}>code</a>, checkpoints and MMBench2 data. This page bundles its data and makes no network requests.
            </p>
            <h4>Use of AI tools</h4>
            <p>Claude (Anthropic) was used throughout: to read the paper and its code, write and test the analysis pipeline, run and check the analyses, write the explanatory document, and build this site. The pre-registered guesses are the author's own, written before any data. The research questions and the decisions recorded in the protocol were worked out in conversation with Claude and decided by the author. Every number on this page is computed by the code in the repository.</p>
          </div>
        </footer>
      </main>
    </>
  );
}
