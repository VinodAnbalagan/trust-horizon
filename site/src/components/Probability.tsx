import { useEffect, useRef } from "react";
import katex from "katex";

/** A formula drawn as MathML, which browsers render without any stylesheet. */
export function M({ tex, block }: { tex: string; block?: boolean }) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (ref.current) katex.render(tex, ref.current, { displayMode: !!block, throwOnError: false, output: "mathml" });
  }, [tex, block]);
  return <span ref={ref} className={block ? "math-block" : "math"} />;
}

const CARDS: { lesson: string; title: string; tex: string; plain: React.ReactNode }[] = [
  {
    lesson: "Random variables",
    title: "Each step is a Bernoulli",
    tex: String.raw`B_t = \begin{cases} 1 & \text{dream broken at step } t \\ 0 & \text{still sound} \end{cases}`,
    plain: <>Like a coin flip that, once it lands on broken, stays broken. Everything below is about predicting these coins.</>,
  },
  {
    lesson: "Continuous random variables",
    title: "What counts as broken",
    tex: String.raw`\tilde e_t \ge \tfrac12\, c, \qquad c = \operatorname{median}_{i \ne j} \lVert z_i - z_j \rVert`,
    plain: <>The distance between two random start states of a task is a continuous random variable; its median <i>c</i> is the chance error. A dream breaks once its smoothed error <i>ẽ<sub>t</sub></i> stays above half of <i>c</i> for 5 steps in a row.</>,
  },
  {
    lesson: "Independence",
    title: "Two independent seeds",
    tex: String.raw`u_s = \operatorname{spread}\big(\hat z^{(1)}_t,\ \hat z^{(2)}_t\big)`,
    plain: <>Dream the next frame twice from independent random noise. If the model is sure, the two draws agree; if it is guessing, they scatter.</>,
  },
  {
    lesson: "Conditional probability",
    title: "The forecast is a conditional",
    tex: String.raw`f_t = P\big(\text{break in } t{+}1..t{+}8 \;\big|\; B_t = 0,\ x_t\big)`,
    plain: <>Given that the dream is still sound at step <i>t</i>, and given what the signals <i>x<sub>t</sub></i> say. Once <i>B<sub>t</sub></i> = 1 the condition fails, which is why the forecast line stops at the break.</>,
  },
  {
    lesson: "Logistic regression",
    title: "Both probabilities are logistic",
    tex: String.raw`p_t = \sigma(w^\top x_t + b), \quad \sigma(z) = \frac{1}{1+e^{-z}}`,
    plain: <><i>x<sub>t</sub></i> holds 11 numbers known at step <i>t</i>: level, recent mean and trend of each signal, scene motion and the step number. One model for the forecast, one for the alarm.</>,
  },
  {
    lesson: "Maximum likelihood",
    title: "Weights that make the data likely",
    tex: String.raw`\hat w = \arg\max_w \sum_t \log P(y_t \mid x_t;\, w)`,
    plain: <>Choose the weights under which the 320 training dreams are most likely, with a small penalty that keeps the weights from chasing noise.</>,
  },
  {
    lesson: "The train test split",
    title: "Three piles of dreams",
    tex: String.raw`\underbrace{320}_{\text{fit}} \;\big|\; \underbrace{160}_{\text{test}} \;\big|\; \underbrace{160}_{\text{unseen, sealed}}`,
    plain: <>Weights come only from the first pile. Every result on this page is from the second, or from the third, which was opened once at the very end.</>,
  },
  {
    lesson: "Conditional probability",
    title: "Survival multiplies conditionals",
    tex: String.raw`S(t) = \prod_{s \le t} \Big(1 - \frac{d_s}{n_s}\Big)`,
    plain: <>Of the <i>n<sub>s</sub></i> dreams still sound and still watched at step <i>s</i>, <i>d<sub>s</sub></i> break there. Chaining "survived this step, given the last" gives the Kaplan-Meier trust horizon.</>,
  },
  {
    lesson: "Probability",
    title: "AUROC is a probability too",
    tex: String.raw`\text{AUROC} = P\big(p_{\text{broken}} > p_{\text{sound}}\big)`,
    plain: <>Pick one broken and one sound step at random: how often does the model give the broken one the higher chance? 0.5 is a coin flip.</>,
  },
];

export function ProbabilityCards() {
  return (
    <div className="pcards">
      {CARDS.map((c) => (
        <div key={c.title} className="pcard">
          <span className="lesson">From class: {c.lesson}</span>
          <h4>{c.title}</h4>
          <M tex={c.tex} block />
          <p>{c.plain}</p>
        </div>
      ))}
    </div>
  );
}
