# Weight-Space Programming Replication

A replication of [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network
weights*](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, Khattab; MIT CSAIL,
November 2026), running its eight experiments with **Claude Opus 5.5** in place of GPT-6 Astra.

**Status:** setting up. No results yet.

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
| 1 | Which game was this network trained on? | high | planned |
| 2 | What does this diffusion model generate? | high | planned |
| 3 | Which language did this model learn, from its middle blocks? | xhigh | planned |
| 4 | Fix a failing car network by editing its weights | high | planned |
| 5 | Write MinAtar networks from scratch | xhigh | planned |
| 6 | Write a 1.5M-weight Go network | xhigh | planned |
| 7 | Which language model is sycophantic? | xhigh | planned |
| 8 | Make Qwen3.5-0.8B better at math | max | planned |

## Setup

```bash
git clone --recurse-submodules --shallow-submodules https://github.com/MatinKhajavi/weight-space-programming-replication.git
cd weight-space-programming-replication
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
