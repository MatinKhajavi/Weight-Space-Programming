# Weight-Space Programming Replication

A replication of [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network
weights*](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, Khattab; MIT CSAIL,
November 2026), running its eight experiments with **Claude Opus 5.5** in place of GPT-6 Astra.

**Status:** experiments 1-5 and exp6 with 0 passes done for Claude Opus 5.5 (exp3: seven conversations, the
rest one sample per item); exp6 with passes, 7 and 8 in progress.

## What this is

The authors released complete logs of their experiments in
[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs): every prompt, every reply, the networks
and the evaluation behind each number. It is pinned here as the `upstream/` submodule. They did not release their
agent harness, the forward-pass server, or most of the environment and scoring code.

This repo:

1. rebuilds the missing pieces;
2. checks them against Astra's released artifacts, reproducing the original numbers before anything else;
3. runs Claude Opus 5.5 on the same prompts with a matched setup.

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
| 6 | Write a 1.5M-weight Go network | xhigh | 0 passes done |
| 7 | Which language model is sycophantic? | xhigh | planned |
| 8 | Make Qwen3.5-0.8B better at math | max | planned |

## Results so far (Claude Opus 5.5, one sample per item)

| Experiment | Opus 5.5 | GPT-6 Astra (published) |
|---|---|---|
| 1. Which game | 4/4 correct (confidence 65-85 %) | 4/4 (96-99 %) |
| 2. Which shape | 3/3 correct (45-60 %) | 3/3 (77-88 %) |
| 3. Which language (7 conversations: 3 Python, 2 romaji, 2 English) | corpus right in 4 of 7 (never romaji); copying heads all found in 6, none named in 1‡; 2 extra | 6 of 6; all found, none extra |
| 4. Fix the car | 1,024/1,024 successes, 1.7 % of weights changed | 1,024/1,024, 3.5 % |
| 5. Breakout | 483.6\* | 517.4 (506.5 in our evaluator) |
| 5. Seaquest | 11.4\*\* | 53.1 (52.0) |
| 5. Asterix | 47.0 | 53.2 (53.2) |
| 6. Go, 0 passes: vs the released same-size PPO networks s0 / s1 / s2 | 53.2 / 40.5 / 54.1 % | 42.6 / 26.3 / 36.3 % (43.2 / 27.0 / 38.8) |

PPO baselines for exp5 (published): Breakout 15.8 / 25.3, Seaquest 5.6 / 88.9, Asterix 25.8 / 55.2 at 100M / 500M
steps. Opus beats the 100M-step PPO networks in all three games, as Astra does.

\* The reply ran past the 128K output cap and Claude Code resumed it once; the reply as it was when cut scores 0.5.
\*\* The cap was reached while still thinking; the answer comes from a second attempt that started over after Claude
Code's "Output token limit hit ... Break remaining work into smaller pieces" message; the cut reply scores 0.1. See
[docs/protocol.md](docs/protocol.md), deviation 5.
‡ Python s3 ran no code: its reply wrote made-up tool calls and outputs as text, and its answer rests on them
(docs/protocol.md, exp3 run notes). Under the original rules it counts as a sample.

## Setup

```bash
git clone --recurse-submodules --shallow-submodules https://github.com/MatinKhajavi/Weight-Space-Programming.git
cd Weight-Space-Programming
cp .env.example .env    # then add your ANTHROPIC_API_KEY
```

`.env` is gitignored. The runner loads it with priority over any key already set in your shell.

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
