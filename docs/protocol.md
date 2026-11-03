# Protocol

How we run Claude Opus 5.5 on the experiments of *Sparks of Weight-Space Programming*: which setting comes from where,
and every known difference from the original.

**Sources.** *blog* = the post (https://rikiyat.github.io/blog/weight-space-programming/). *repo* = the authors' logs,
pinned here as `upstream/` (RikiyaT/lm_understand-logs @ `4d09ba6`): mostly its `README.md` files and the per-turn
`run.json` records. The blog states the experiment setups but not how the model was called; everything about the
model call comes from the repo.

## Model call

| Setting | Original (Astra) | Source | Ours |
|---|---|---|---|
| Model | `gpt-6-astra` ("GPT-6 Astra") | blog, repo README | `claude-opus-5-5` |
| Interface | Codex CLI: one `codex exec` per turn, `codex exec resume <thread>` to continue, prompt on stdin | repo README, `run.json` `cmd` | Claude Agent SDK (Python): one `query()` per turn, `resume=<session_id>` to continue |
| Built-in tools | all disabled; every turn's event stream audited tool-free | `harness/codex_config.toml`, repo README | `tools=[]`; every turn audited for tool calls |
| System prompt | `Answer the user's request.` | `harness/instructions.md` | the same line, preceded by Claude Code's fixed line (see Deviations) |
| Reasoning effort | `high`: exps 1, 2, 4. `xhigh`: exps 3, 5, 6, 7. `max`: exp 8 | `run.json` (not stated in the blog) | the same level names |
| Reasoning summaries | `detailed` in exps 1, 2, 5, 7; none in 3, 4, 6, 8 | `run.json` | `summarized` in every experiment |
| Context and compaction | `model_context_window=872000`, `model_auto_compact_token_limit=1000000` (above the window, so it never triggers) | `run.json` | 1M window, `DISABLE_AUTO_COMPACT=1` |
| Sampling | no temperature or seed set in any of the 1,125 calls | `run.json` | none (Opus 5.5 does not accept one) |
| Max output tokens | not recorded | none | 128,000 |
| Model fallback | none | none | none |
| Prompt text | `prompt.txt` / `message.txt` per run | repo | sent byte for byte |
| Samples per item | 1 | repo runs; blog ("in a single try") | 1 first; up to 4 more added later as samples 2-5 |

Each run is stored in `runs/<model>/<exp>/<item>/s<sample>/`. The logs of the first two runs (exp1 and exp5 Breakout) were filtered afterwards in the same way the runner now filters every log: streaming progress events and the opaque thinking signatures are dropped, and later record fields were added from the logs.

## Isolation of the model call

Claude Code adds context of its own unless told not to. With SDK defaults, a request carries `CLAUDE.md` files,
auto-memory, `@path` file contents, about ten `<system-reminder>` blocks, today's date and 12 tools, at effort
`medium`. Our configuration:

```python
ClaudeAgentOptions(
    model="claude-opus-5-5",
    system_prompt="Answer the user's request.",
    tools=[],                      # no built-in tools
    setting_sources=[],            # no user/project/local settings, no CLAUDE.md
    strict_mcp_config=True,        # no MCP servers
    verbatim_prompts=True,         # no @path expansion, no slash commands, no per-turn attachments
    extra_args={"bare": None},     # --bare: no hooks, plugins, auto-memory
    effort=...,                    # per experiment, see above
    thinking={"type": "adaptive", "display": "summarized"},
    max_buffer_size=16 * 1024 * 1024,   # default is 1 MB; the exp4 prompt alone is 0.58 MB
    cwd=<empty run folder>,
    env={"CLAUDE_CONFIG_DIR": <fresh folder per run>, "DISABLE_AUTO_COMPACT": "1",
         "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000", "CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS": "1",
         "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "ANTHROPIC_API_KEY": <from .env>},
)
```

The runner also removes every inherited `CLAUDE*` / `ANTHROPIC*` environment variable before starting the SDK, because
the SDK passes its parent's environment to Claude Code (a parent Claude Code session sets `CLAUDE_EFFORT`, among
others).

**How this was verified.** The SDK was pointed at a local stand-in for the API (dummy key, `ANTHROPIC_BASE_URL` on
localhost) that records every request body. Marker strings were planted in a `CLAUDE.md`, an auto-memory file and an
`@`-mentioned file. With SDK defaults all three reached the request; with the configuration above none did, the user
message was the prompt byte for byte, no tools were sent, effort and thinking were as set, and a resumed second turn
re-sent the history unchanged. This check is local tooling and is not part of the repo. Each run also records the
prompt's token count next to the API's reported input tokens, so injected context would show up as a gap.

## Known deviations from the original

1. **System prompt.** Claude Code sends two blocks before the original line: a billing-header line
   (`x-anthropic-billing-header: ...`; we don't know whether the model sees it) and
   `You are a Claude agent, built on Anthropic's Claude Agent SDK.` The original Codex setup also added context of its
   own, about 3.5K tokens judging from the reported token counts, with contents not released. A plain-API cross-check on
   exp1 will show whether the extra sentence matters.
2. **Reasoning summaries** are requested in every experiment, not only exps 1, 2, 5, 7. Per Anthropic's documentation,
   this changes what is returned, not the reasoning itself.
3. **Effort levels** have the same names on both models but are not calibrated to each other.
4. **Output cap** is 128K tokens. The original cap is not recorded; Astra's longest reply was about 29K output tokens.
5. **Replies longer than the output cap (marked \*).** When a reply hits the 128K cap, Claude Code sends
   `Output token limit hit. Resume directly — no apology, no recap of what you were doing. Pick up mid-thought if that
   is where the cut happened. Break remaining work into smaller pieces.` and the model finishes in a second message.
   No setting turns this off. The message carries no information about the task, but the prompts say "This is a single
   response" and the original never met this case: no output cap was set in any of their calls, and Astra's longest
   reply was about 29K tokens. We score the completed answer and mark it \*; the reply as it was when cut is kept in
   `answer_cut.md` and its score is recorded in `score.json` under `cut_at_output_cap`. The message carries no
   information about the game or the score, but it does tell the model it ran out of space and to work in smaller
   pieces. When the cut reply had written text, the second request carries the whole cut reply, thinking included
   (exp5 Breakout: the second reply repeats the first 75 statements and finishes the block). When the cap was reached
   while still thinking, the second request carries only the prompt and the message, so the model starts over with
   that instruction (exp5 Seaquest, whose second attempt refers to its "compact output budget"); such a result is a
   second attempt under a different instruction, not a completed reply.
6. **Tool calls written as text.** With no tools available, a reply can contain made-up tool calls and made-up
   outputs written as text (`{"tool":"code",...}` / `{"tool":"code_output",...}`) instead of the protocol's code block.
   Each reply's count is recorded as `fake_tool_calls` in its `run.json`. Under the original rules a reply without a
   code block is the final answer.

## Retries (from the original logs)

The rules below are observed in the authors' harness logs, not documented by them.

- Failed call: retry after 60, 120, 240, then 480 s.
- Usage or rate limit: retry after 900, 1,350, 2,025, then 3,038 s.
- Empty final reply: discard it, wait 60 s, and send the same first message again in a new conversation (exp3
  `english_r0`).
- Ours only: a request the API rejects (a 4xx error other than 429) stops the run instead of being retried.

## Scoring

- Each scorer is validated on Astra's released answers, where it must reproduce the original scores, and is frozen
  before any Opus answer is scored.
- Where the original rule is not released (the exp1 and exp2 keyword rules), ours is written down in the scorer and
  agreed before use.
- Each environment and evaluator is checked against Astra's released artifacts before it scores any Opus answer, for
  example the exp4 base network at 0/256 and the exp5 networks at about 517 / 53 / 53.
- Where the original seeds are not recorded (exps 4, 5), comparisons are statistical, not exact.

### Validation so far

- **exp1, exp2 scorers** (`wsp/scoring.py`): on Astra's 7 released answers they give the same mapped answer, the same
  right/wrong and the same confidence as the released `eval/scores.json` (4/4 games, 3/3 shapes). Our keyword rules:
  names only (a description without a name is wrong); two different labels named is wrong ("hedged"); whole-word
  matching; exp2 scores the shape only, not filled vs outline.
- **exp5 edit reader** (`wsp/exp5.py`): rebuilds Astra's three released networks bit for bit, no statement skipped.
- **exp5 evaluator**: official MinAtar 1.0.15 `Environment`, one environment per worker with episodes back to back
  (its last action carries over between episodes, as in the official wrapper). A seed s gives 10 workers, worker w
  with one environment seeded s + w that plays its 100 episodes back to back, so seeds less than 10 apart share
  streams. Astra's Breakout network scores 506.5 ± 7.6 (seed 0) and 508.1 ± 7.6 (seed 1000) against the released
  517.4 ± 7.4: about 1 standard error low in both blocks. Astra's and Opus's networks are scored with the same evaluator and seeds, so a
  small offset between our evaluator and theirs does not affect the comparison. Seaquest: 52.0 ± 1.4 (released
  53.1 ± 1.4). Asterix: 53.2 ± 1.0 (released 53.2 ± 1.0).
- **exp4 edit reader and evaluator** (`wsp/exp4.py`): applying Astra's 51 statements to the network as shown rebuilds
  the released `edited_actor.npz` bit for bit (6,073 entries changed). The evaluator is Kinetix at the original
  commit with the original training config, run on Modal. The base network as shown gives success 0/256, mean return
  0.15191, progress 0.33900, most-likely-action episode a timeout (released: 0/256, 0.15190, 0.33899, timeout);
  Astra's edit gives 1,024/1,024 successes in 73 steps, return 1.37385 (released: the same), and its sampled episode
  matches the trace printed in the prompt at every logged step. Progress is 1 - (closest distance between the green
  wheel and the blue box) / (distance at the start), 1 on success; this definition reproduces the released value.
  Runs use Modal CPUs for now (`gpu=None`), because GPUs need a payment method on the Modal workspace.

## Order of work

1. exp1 Breakout and exp5 Breakout first, to confirm the setup end to end before the rest.
2. The rest of exps 1, 2 and 5, one sample each.
3. The exp4 model call; scored once the Kinetix evaluation reproduces the original base result.
4. The multi-turn experiments (3, 6, 7, 8). Their harness gaps (listed below) get designed and written down here
   before any run.

## Multi-turn experiments

One Agent SDK call per turn, each resuming the previous one, as the original resumed Codex with
`codex exec resume <thread>`. Turn 1 is the task message; each later turn is the output of the code block in the
previous reply, in the original's template; a reply without a code block is the final answer.

A failed turn is retried with the same waits as above, after rewinding the conversation to the last good reply, so the
model never sees the failed attempt.

**exp3.** Task message: upstream `exp3_language/runs/python_r0/message.txt` (identical for all six original
conversations). Sandbox: a fresh Docker container per run (`sandbox/Dockerfile`, no network), with `np` and `W` (the
40 block arrays from `networks/<language>/blocks.npz`) preloaded; 600 s per run; output cut at 20,000 characters;
stderr's last 6,000 characters appended, as in upstream `harness/sandbox.py`; at most 100 runs. Output template:
`OUTPUT of run <k> of at most 100:` followed by the output in a code fence. Our choices where the original behaviour is
not in the logs: a reply with several code blocks runs the first (logged as `blocks_in_reply`); after 100 runs the
model is asked for its final answer. The ban on forward passes is checked afterwards by reading every code run, as the
authors did.

**exp3 run notes.**
- Python s3 (the first of the seven starts) ran no code. In its first reply the model wrote about 63 made-up tool
  calls with made-up outputs as text (`{"tool":"code",...}` / `{"tool":"code_output",...}`), ran past the output cap
  and was resumed, and its final answer rests on those made-up outputs. Under the original rules a reply without a
  code block is the final answer, so it counts as a sample (corpus python, order shuffled, no copying head named),
  marked ‡. The other six conversations were started after it.
- The ban on forward passes was checked by reading all 232 code runs: none runs the network or a layer on an input.
  One borderline run (Python s2, run 45) applies a ReLU to z-scored layer-0 MLP pre-activations of directions
  estimated from the weights; it is not the network's own computation (no LayerNorm, bias or output projection), and the
  model judged its result to be noise.
- In these six conversations the per-turn costs in `run.json` were converted afterwards from Claude Code's cumulative
  session cost (kept as `session_cost_usd`) by the same rule the runner now applies itself.

**exp6.** Task message: upstream `exp6_go_from_scratch/runs/budget_<n>/message.txt`. Sandbox: the same Docker
image with `/work` kept between runs and the prelude taken verbatim from upstream `harness/sandbox.py` (`SHAPES`,
`NAMES`, `load`, `save`, `saved`, `inp`, `mv`, `PASS`); 900 s per run; output cut at 20,000 characters; at most 120
Python runs. Memory is 7 GB, not the original 96 GB. Output template: `OUTPUT of Python run <k> of at most 120:`. Only
budget 0 (no pass tools) is implemented so far; our choices: a pass block in a budget-0 session gets "There are no
pass tools in this session (budget 0)."; after 120 runs the model is asked for its final answer.

Games (`wsp/exp6.py`): pgx 2.6.0 `go_9x9`, whose rules are the task's as they stand (no changes); input from pgx's
observation planes 0, 1 and 16. Checked against the published head-to-heads of Astra's released networks with the
released same-size PPO networks, 2,000 games each: budget 0 vs seeds 0 / 1 / 2: 43.2 / 27.0 / 38.8 % (published
42.6 / 26.3 / 36.3); budget 50: 73.0 / 69.2 / 80.9 % (74.2 / 70.0 / 78.5); game endings, margins and the results
against a random player also match within noise. A written network is scored against the three released same-size
PPO networks and a random player, 2,000 games each. The practice and hidden opponents of the original are not
released and are not rebuilt yet.

**exp6 run notes (budget 0).** The 9 code runs and the model's design scripts in `work/` only build weights from
formulas and compare arrays; none runs the network, trains or searches. About 5 % of its games (80-95 of 2,000 per
opponent) are lost by its own repeating move, mostly a three-move cycle: it plays into a two-stone self-atari, the
opponent captures, and its capture bonus makes it retake one stone, which repeats the earlier board.

## Open: multi-turn harness gaps

Never observed in the original logs, so the original behaviour is unknown:

- exp6: the reminder text sent when the model answers with passes left.
- exps 6, 7, 8: handling of a reply with several code blocks; behaviour at a run or pass limit; the message for a
  failed request.
