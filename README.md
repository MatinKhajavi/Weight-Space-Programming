# Weight-Space Programming Replication

A replication of [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network
weights*](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, Khattab; MIT CSAIL,
November 2026), running its eight experiments with **Claude Opus 5.5** and **Claude Fable 5.1** in place of GPT-6 Astra.

**Status:** all eight experiments done for Claude Opus 5.5, experiments 1-5 for Claude Fable 5.1.

## What this is

The authors released complete logs of their experiments in
[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs): every prompt, every reply, the networks
and the evaluation behind each number. It is pinned here as the `upstream/` submodule. They did not release their
agent harness, the forward-pass server, or most of the environment and scoring code.

This repo:

1. rebuilds the missing pieces;
2. checks them against Astra's released artifacts, reproducing the original numbers before anything else;
3. runs Claude models on the same prompts with a matched setup.

Every setting, where it comes from, and every known deviation from the original are in
[docs/protocol.md](docs/protocol.md).

## Experiments

| # | Experiment | Original effort | Status |
|---|---|---|---|
| 1 | Which game was this network trained on? | high | done |
| 2 | What does this diffusion model generate? | high | done |
| 3 | Which language did this model learn, from its middle blocks? | xhigh | done |
| 4 | Fix a failing car network by editing its weights | high | done |
| 5 | Write MinAtar networks from scratch | xhigh | done |
| 6 | Write a 1.5M-weight Go network | xhigh | done |
| 7 | Which language model is sycophantic? | xhigh | done |
| 8 | Make Qwen3.5-0.8B better at math | max | done |

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/results-dark.png">
  <img alt="Bar charts of the eight experiments: Claude Opus 5.5, Claude Fable 5.1 and GPT-6 Astra" src="docs/figures/results.png">
</picture>

### Experiments 1-5

| Experiment | Claude Opus 5.5 | Claude Fable 5.1 | GPT-6 Astra |
|---|---|---|---|
| 1. Which game | 4/4 | 3/4<sup>1</sup> | 4/4 |
| 2. Which shape | 3/3 | 2/3 (read the square as a bar) | 3/3 |
| 3. Which language: corpus named right | 4 of 7<sup>2</sup> | 4 of 6 | 6 of 6 |
| 3. Copying heads found | all in 6 of 7, 2 extra | all in 5 of 6, 1 extra | all, none extra |
| 4. Fix the car | fixed, 1.7 % of weights changed | fixed, 96 % changed (rebuilt from zero) | fixed, 3.5 % changed |
| 5. Breakout | 483.6<sup>3</sup> | 38.4<sup>3</sup> | 517.4 |
| 5. Seaquest | 11.4<sup>4</sup> | 50.8<sup>3</sup> | 53.1 |
| 5. Asterix | 47.0 | 49.4 | 53.2 |

- Exps 1 and 2, confidence in the answers: Opus 65-85 % and 45-60 %, Fable 55-75 % and 60-68 %, Astra 96-99 % and
  77-88 %.
- Exp 3: Opus had 3 conversations on the Python network and 2 on each other network; Fable 2 on each.
- Exp 5: Astra's networks score 506.5 / 52.0 / 53.2 in our evaluator. Published PPO baselines: Breakout 15.8 / 25.3,
  Seaquest 5.6 / 88.9, Asterix 25.8 / 55.2 at 100M / 500M steps. Fable's Breakout network misses the ball by one
  column: it reaches the landing column at the last moment, so one sticky action (the game repeats the previous move
  10 % of the time) is enough to miss.
- <sup>1</sup> Pong is scored wrong because the answer names two games ("Single-player Pong / squash (Breakout-style
  paddle vs. wall)"), under the rule fixed before any answer was scored. Read by hand, the answer is Pong.
- <sup>2</sup> One conversation (Python s3) ran no code: it wrote made-up tool calls and outputs as text and answered
  from them. Under the original rules it still counts.
- <sup>3</sup> The reply ran past the 128K output cap and Claude Code resumed it; the score is the completed reply. The
  cut reply alone scores 0.5 on Breakout and 0.1 on Seaquest.
- <sup>4</sup> The cap was reached while still thinking, so the answer is a second attempt that started over after
  Claude Code's "Output token limit hit ... Break remaining work into smaller pieces" message. The cut reply scores 0.1
  ([docs/protocol.md](docs/protocol.md), deviation 5).

### Experiment 6: write a Go network (win rate %)

| Passes | Opus vs released PPO s0 / s1 / s2 | Astra vs released PPO | Opus vs hidden PPO s0 / s2 | Astra vs hidden PPO |
|---|---|---|---|---|
| 0 | 53.3 / 40.5 / 54.2 | 42.6 / 26.3 / 36.3 | 66.5 / 52.0 | 54.1 / 54.9 |
| 10 | 72.0 / 56.6 / 74.8 | not published | 83.6 / 78.6 | 72.9 / 60.0 |
| 25 | 43.3 / 31.3 / 49.2 | not published | 55.8 / 41.5 | 76.0 / 62.3 |
| 50 | 44.1 / 34.9 / 44.1 | 74.2 / 70.0 / 78.5 | 70.0 / 46.6 | 84.0 / 80.8 |

- The released same-size PPO networks are the like-for-like comparison. Against them, Astra's networks score 43.2 /
  27.0 / 38.8 (0 passes) and 73.1 / 69.3 / 80.9 (50 passes) in our evaluator.
- The hidden opponents were not released. They were rebuilt so that Astra's four networks score as published against
  them (RMS 3.6 points; [docs/protocol.md](docs/protocol.md)).

### Experiment 7: which model is sycophantic (P of the right answer, pairs 1 / 2 / 3)

| Round | Opus 1 | Opus 2 | Opus 3 | Astra 1 | Astra 2 | Astra 3 |
|---|---|---|---|---|---|---|
| Attention tool | 0.94<sup>5</sup> | 0.97 | 0.93 | 0.90 | 0.97 | 0.55 |
| Weights only, round 1 (18 layers) | 0.32 | 0.30 | not counted<sup>6</sup> | 0.50 | 0.50 | 0.50 |
| Weights only, round 2 | 0.28 | 0.75 | not counted<sup>6</sup> | 0.50 | 0.50 | 0.50 |
| Weights only, round 3 | 0.30<sup>5</sup> | 0.45 | 0.30<sup>5</sup> | 0.50 | 0.50 | 0.50 |
| Weights only, round 4 (full weights) | 0.73 | 0.68<sup>5</sup> | 0.08 | 0.50<sup>7</sup> | 0.55 | 0.30<sup>7</sup> |

- Above 0.5 is a right call. Astra answered 0.5 in rounds 1-3, naming A as a tie-break.
- <sup>5</sup> Some runs broke the ban on running the model (mostly q/k normalisation of constructed vectors; round 4
  pair 2 also normalised layer-1 outputs and ran part of layer 1). The answer rests on other, clean runs, so it counts.
- <sup>6</sup> The answer (right in both cases) rests on runs that broke the ban, so it is not counted.
- <sup>7</sup> The authors stopped Astra's round 4 by hand after 29-30 runs, with only pair 2 finished; pairs 1 and 3
  show its last answer before that.
- The model pairs are rebuilt with the authors' scripts: same release IDs and answer keys, behaviour within noise.

### Experiment 8: make Qwen3.5-0.8B better at math (accuracy %)

| Passes | Opus MATH-500 | Opus GSM8K | Opus IFEval | Astra MATH-500 | Astra GSM8K | Astra IFEval |
|---|---|---|---|---|---|---|
| 0 (unedited) | 8.8 | 12.8 | 22.0 | 7.9 | 13.8 | 21.0 |
| 10 | 56.2 | 64.5 | 55.5 | 12.3 | 25.7 | 49.0 |
| 25 | 33.3 | 50.5 | 36.2 | 14.1 | 25.1 | 39.9 |
| 50 | 54.4 | 66.3 | 50.4 | 48.1 | 67.1 | 62.3 |

- Astra's numbers are the published ones. Opus's are our evaluation: the authors' `evaluate.py` on MATH-500 (4 samples
  per problem), GSM8K (2) and IFEval (2), which scores Astra's 50-pass edit at 50.8 / 67.4 / 62.5.
- What Opus changed: with 10 passes, one MLP neuron zeroed, one x4 and eight attention heads halved; with 25, three
  attention outputs reversed in the span of eight line-opening words' unembedding rows; with 50, three attention heads
  zeroed and one MLP neuron x2.
- The edited models are on Hugging Face:
  [mtkhajavi/weight-space-math-edits-by-claude-opus-55](https://huggingface.co/collections/mtkhajavi/weight-space-math-edits-by-claude-opus-55-6aebde9378a51a6c7d803eeb).

## Setup

```bash
git clone --recurse-submodules --shallow-submodules https://github.com/MatinKhajavi/Weight-Space-Programming.git
cd Weight-Space-Programming
cp .env.example .env    # then add your ANTHROPIC_API_KEY
```

`.env` is gitignored. The runner loads it with priority over any key already set in your shell. Exps 4 and 6-8 run
parts on Modal (`MODAL_PROFILE`); exps 7 and 8 need `HF_TOKEN` (Gemma 3 and Qwen3.5 weights). The sandboxes are
Docker images: `docker build -t wsp-sandbox sandbox` and `docker build -t wsp-sandbox-numpy -f sandbox/numpy.Dockerfile
sandbox`.

## Credit

The experiment designs, prompts, networks and original results are the work of Rikiya Takehi, Jayden Teoh, Alex Zhang
and Omar Khattab. If you use this replication, please cite their post:

```bibtex
@misc{takehi2026weightspace,
  title        = {{Sparks of Weight-Space Programming: LLMs can natively read \& write neural network weights}},
  author       = {Takehi, Rikiya and Teoh, Jayden and Zhang, Alex and Khattab, Omar},
  year         = {2026},
  month        = nov,
  howpublished = {\url{https://rikiyat.github.io/blog/weight-space-programming/}},
  note         = {Blog post}
}
```

## License

Our code is MIT. Our run logs and results are CC BY 4.0. Logs of exp5 quote MinAtar source and carry its GPL-3.0
license. Upstream content keeps its own licenses. Details in [LICENSE](LICENSE).
