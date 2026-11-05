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
| Samples per item | 1 | repo runs; blog ("in a single try") | 1 |

Each run is stored in `runs/<model>/<exp>/<item>/s<sample>/`. The logs of the first two runs (exp1 and exp5 Breakout) were filtered afterwards in the same way the runner now filters every log: streaming progress events and the opaque thinking signatures are dropped, and later record fields were added from the logs.

In the multi-turn experiments (3, 6, 7, 8) each turn is one Agent SDK call resuming the previous one, as the
original resumed Codex with `codex exec resume <thread>`. Turn 1 is the task message; each later turn is the output
of the code block in the previous reply, in the original's template; a reply without a code block is the final
answer.

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
re-sent the history unchanged. The check is `checks/capture.py`. Another local check compares
each prompt's token count with the API's reported input tokens, so injected context would show up as a gap; run
records written before 2026-11-03 also carry that count as `prompt_tokens`.

## Known deviations from the original

1. **System prompt.** Claude Code sends two blocks before the original line: a billing-header line
   (`x-anthropic-billing-header: ...`; we don't know whether the model sees it) and
   `You are a Claude agent, built on Anthropic's Claude Agent SDK.` The original Codex setup also added context of its
   own, about 3.5K tokens judging from the reported token counts, with contents not released. Whether the extra sentence
   matters is not tested.
2. **Reasoning summaries** are requested in every experiment, not only exps 1, 2, 5, 7. Per Anthropic's documentation,
   this changes what is returned, not the reasoning itself.
3. **Effort levels** have the same names on both models but are not calibrated to each other.
4. **Output cap** is 128K tokens. The original cap is not recorded; Astra's longest reply was about 29K output tokens.
5. **Replies longer than the output cap.** When a reply hits the 128K cap, Claude Code sends
   `Output token limit hit. Resume directly — no apology, no recap of what you were doing. Pick up mid-thought if that
   is where the cut happened. Break remaining work into smaller pieces.` and the model finishes in a second message.
   No setting turns this off. The message carries no information about the task, but the prompts say "This is a single
   response" and the original never met this case: no output cap was set in any of their calls, and Astra's longest
   reply was about 29K tokens. We score the completed answer (flagged in the README); the reply as it was when cut is kept in
   `answer_cut.md` and its score is recorded in `score.json` under `cut_at_output_cap`. The message carries no
   information about the game or the score, but it does tell the model it ran out of space and to work in smaller
   pieces. When the cut reply had written text, the second request carries the whole cut reply, thinking included
   (exp5 Breakout: the second reply repeats the first 75 statements and finishes the block). When the cap was reached
   while still thinking, the second request carries only the prompt and the message, so the model starts over with
   that instruction (exp5 Seaquest, whose second attempt refers to its "compact output budget"); such a result is a
   second attempt under a different instruction, not a completed reply. The answer that is scored is the last message's
   text (Claude Code's final result). Claude Code also resumes after a reply is cut off mid-stream, with
   `Your response above was cut off mid-stream. Resume directly from where it stops — no apology, no recap. If none of
   it survived, answer the request from the start.` (Fable exp3 Python s2, turn 12: the cut reply had no text, and the
   reply after it repeated the same code block); `resumed_after_output_cap` and `injected_messages` record both cases.
   A third message follows a reply stopped by the safety classifier: `Your response above was stopped by a safety
   classifier — this is not a tool or API error. The rest of it was withheld, and tool calls in it that had not finished
   did not run. Do not produce that content again, even reworded.` (exp6 budget 50 turns 1 and 20; exp8 p10c turns 64
   and 79, p25c turns 20, 38 and 40).
   For Fable 5.1 Claude Code records `maxOutputTokens` 64,000, but its replies were cut at about 128,000 tokens.
6. **Tool calls written as text.** With no tools available, a reply can contain made-up tool calls and made-up
   outputs written as text (`{"tool":"code",...}` / `{"tool":"code_output",...}`) instead of the protocol's code block.
   A local check counts them in every reply; run records written before 2026-11-03 also carry the count as
   `fake_tool_calls`. Under the original rules a reply without a code block is the final answer.

## Retries (from the original logs)

The rules below are observed in the authors' harness logs, not documented by them.

- Failed call: retry after 60, 120, 240, then 480 s.
- Usage or rate limit: retry after 900, 1,350, 2,025, then 3,038 s.
- Empty final reply: discard it, wait 60 s, and send the same first message again in a new conversation (exp3
  `english_r0`).
- Ours only: a request the API rejects (a 4xx error other than 429) stops the run instead of being retried.
- In a multi-turn experiment, a failed turn is retried with the same waits after rewinding the conversation to the
  last good reply, so the model never sees the failed attempt.

## Scoring

- Each scorer is validated on Astra's released answers, where it must reproduce the original scores, and is frozen
  before any Opus answer is scored.
- Where the original rule is not released (the exp1 and exp2 keyword rules), ours is written down in the scorer and
  agreed before use.
- Each environment and evaluator is checked against Astra's released artifacts before it scores any Opus answer, for
  example the exp4 base network at 0/256 and the exp5 networks at about 517 / 53 / 53.
- Where the original seeds are not recorded (exps 4, 5), comparisons are statistical, not exact.

### Validation

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
  53.1 ± 1.4). Asterix: 53.2 ± 1.0 (released 53.2 ± 1.0). Seed blocks of the same network differ by up to about 4 standard
  errors of one block (Fable's Asterix network: 49.4 ± 0.9 at seed 0, 45.6-47.4 at three other blocks); reported scores use seed 0.
- **exp4 edit reader and evaluator** (`wsp/exp4.py`): applying Astra's 51 statements to the network as shown rebuilds
  the released `edited_actor.npz` bit for bit (6,073 entries changed). The evaluator is Kinetix at the original
  commit with the original training config, run on Modal. The base network as shown gives success 0/256, mean return
  0.15191, progress 0.33900, most-likely-action episode a timeout (released: 0/256, 0.15190, 0.33899, timeout);
  Astra's edit gives 1,024/1,024 successes in 73 steps, return 1.37385 (released: the same), and its sampled episode
  matches the trace printed in the prompt at every logged step. Progress is 1 - (closest distance between the green
  wheel and the blue box) / (distance at the start), 1 on success; this definition reproduces the released value.
  It runs on Modal CPUs.

## Experiment 3: which language

### Setup

Task message: upstream `exp3_language/runs/python_r0/message.txt` (identical for all six original
conversations). Sandbox: a fresh Docker container per run (`sandbox/Dockerfile`, no network), with `np` and `W` (the
40 block arrays from `networks/<language>/blocks.npz`) preloaded; 600 s per run; output cut at 20,000 characters;
stderr's last 6,000 characters appended, as in upstream `harness/sandbox.py`; at most 100 runs. Output template:
`OUTPUT of run <k> of at most 100:` followed by the output in a code fence. Our choices where the original behaviour is
not in the logs: a reply with several code blocks runs the first (logged as `blocks_in_reply`); after 100 runs the
model is asked for its final answer. The ban on forward passes is checked afterwards by reading every code run, as the
authors did.

### Run notes

- Python s3 (the first of the seven starts) ran no code. In its first reply the model wrote about 63 made-up tool
  calls with made-up outputs as text (`{"tool":"code",...}` / `{"tool":"code_output",...}`), ran past the output cap
  and was resumed, and its final answer rests on those made-up outputs. Under the original rules a reply without a
  code block is the final answer, so it counts as a sample (corpus python, order shuffled, no copying head named),
  flagged in the README. The other six conversations were started after it.
- The ban on forward passes was checked by reading all 232 code runs: none runs the network or a layer on an input.
  One borderline run (Python s2, run 45) applies a ReLU to z-scored layer-0 MLP pre-activations of directions
  estimated from the weights; it is not the network's own computation (no LayerNorm, bias or output projection), and the
  model judged its result to be noise.
- Fable 5.1 (166 code runs read): no forward pass. Two borderline runs of the same kind: English s1 run 14 applies a
  ReLU to layer-0 MLP pre-activations of directions estimated from the weights (the model judged the result
  untrustworthy and did not use it); Python s2 runs 34-35 gate MLP0 neurons by thresholded regression coefficients
  and sum their outputs, a rough token-to-MLP0-to-logit map that part of its corpus reasoning rests on.
- In Opus's six later conversations the per-turn costs in `run.json` were converted afterwards from Claude Code's cumulative
  session cost (kept as `session_cost_usd`) by the same rule the runner now applies itself.

## Experiment 6: write a Go network

### Setup

Task message: upstream `exp6_go_from_scratch/runs/budget_<n>/message.txt`. Sandbox: the same Docker
image with `/work` kept between runs and the prelude taken verbatim from upstream `harness/sandbox.py` (`SHAPES`,
`NAMES`, `load`, `save`, `saved`, `inp`, `mv`, `PASS`); 900 s per run; output cut at 20,000 characters; at most 120
Python runs. Memory is 7 GB, not the original 96 GB. Output template: `OUTPUT of Python run <k> of at most 120:`.

Passes (budgets 10, 25, 50), rebuilt from the logs (`runs/budget_N/requests`, `tool_calls.jsonl`, `work/pass_<k>/`):
one counter for ```` ```games ```` and ```` ```positions ```` passes; the result templates (`Games pass k of N done.`
with the compact JSON, `Passes left: n.`), the files in `/work/pass_<k>/` and their README texts as upstream's. A games
pass plays the current saved network against the practice opponents in name order, Black in the first half against
each; a positions pass reads "board" positions as they stand and replays "moves" positions with the game's rules (on
Modal CPU). Our choices where the logs show nothing: a reply with several blocks runs the first; the texts for a
final answer with passes left, a pass after the last one, a failed request (not counted) and a reply after 120
Python runs; the seed of games pass k is 1000 k (+ j for the j-th opponent); with budget 0 a pass block gets "There
are no pass tools in this session (budget 0)."

Games (`wsp/exp6.py`): pgx 2.6.0 `go_9x9`, whose rules are the task's as they stand (no changes); input from pgx's
observation planes 0, 1 and 16. Checked against the published head-to-heads of Astra's released networks with the
released same-size PPO networks, 2,000 games each: budget 0 vs seeds 0 / 1 / 2: 43.2 / 27.0 / 38.8 % (published
42.6 / 26.3 / 36.3); budget 50: 73.1 / 69.3 / 80.9 % (74.2 / 70.0 / 78.5); game endings, margins and the results
against a random player also match within noise. A written network is scored against the three released same-size
PPO networks and a random player, 2,000 games each.

Opponents (`wsp/exp6_opponents.py`): the practice ("weak", "strong") and hidden (PPO seeds 0 and 2, REINFORCE seeds
0 and 2) opponents are not released and were rebuilt: 243 -> 32 -> 32 -> 82 networks trained on Modal GPUs (A10G,
$14.63 in all) by self-play with half the games against the current network and half against snapshots every 100
iterations (training only against snapshots learned early-pass exploits and cycled). The recipe is ours (upstream's
`gomatch train.py` is not released): Adam, PPO with PureJaxRL-style defaults, REINFORCE without baseline, +1/-1 for
a win/loss. For each role the checkpoint is chosen so that Astra's four released networks score as published against
it: RMS 3.6 points over 23 targets on 2,000 fresh games (ppo_s0 6.3, ppo_s2 3.5, strong 3.0, weak 3.9; REINFORCE within
1 point); the chosen PPO checkpoints are at iterations 9,000-12,500, not 20,000. Held-out check: the released
same-size PPO networks score 12-28 points higher against our PPO opponents than against the originals in five of
six pairings (4 points lower in the sixth), so our
opponents match the originals against hand-written networks, not against PPO-style play; the head-to-heads against
the released same-size PPO networks stay the like-for-like measure. A small opponent is played as an exact 512 x 6
network (zero padding, identity layers 3-6). Scoring: 2,000 games against each hidden opponent and random (seeds
777 + j, upstream's order) and against the three released same-size PPO networks and random (seeds 0 + j).

### Run notes

**Budget 0.** The 9 code runs and the model's design scripts in `work/` only build weights from
formulas and compare arrays; none runs the network, trains or searches. About 5 % of its games (80-95 of 2,000 per
opponent) are lost by its own repeating move, mostly a three-move cycle: it plays into a two-stone self-atari, the
opponent captures, and its capture bonus makes it retake one stone, which repeats the earlier board.

**Budget 10.** All 83 Python runs and the build scripts read: no fitting (every coefficient is a
hand-set literal, changed between versions by comparing games passes), no search (`go.py` is Go-rule code used to
label moves of recorded games), and the network is never run in the sandbox. Borderline: runs 23-26, 38, 43, 44, 73
and 78-81 compute idealised versions of planned features (for example the self-atari sub-circuit, run 73) on the
boards of pass games and score them against exact Go-rule labels, to choose between feature formulas. The final
network is the one measured in the last pass (rebuilt bit for bit from `build11.py`). The final answer credits two
passes of different versions to one and overstates its expected hidden score. Turn 3 ran into the output cap after
2,329 made-up tool calls written as text (`<invoke name="python">`, 132K output tokens); Claude Code resumed it and
nothing ran. The final answer (turn 94) was resumed after a mid-stream cut.

**Budget 25.** All 88 Python runs and the build scripts read: no fitting, no search, the network is
never run in the sandbox; borderline as in budget 10 (runs 20 and 58 compute an idealised "group mass" term on toy
and pass-game boards to set a threshold). Turns 1 and 2 were resumed by Claude Code (output cap; cut mid-stream).
Turn 3, resumed after the output cap, holds four made-up tool calls written
as text (`<invoke name="bash">`); nothing ran, the harness sent its reminder, and run 3 found that the file they would
have written does not exist. The final network is the one measured in the last pass (rebuilt bit for bit). It ended
weaker than budget 10: its liberty evidence kept producing false ataris on big groups, and passes 6-15 went to patches
that did not hold (38-51 % against "strong" until pass 16).

**Budget 50.** All 120 Python runs read: no fitting, no search, the network is never run in the
sandbox; borderline as in budgets 10 and 25 (runs 12-17 score an idealised liberty-evidence rule on pass-game boards
against Go-rule labels, which chose the rule in the final design; runs 62-64 evaluate the firing condition of units
that were later dropped). No made-up tool calls; turns 1 and 20 were stopped once by Opus's safety classifier and
resumed. The final network (v27) is the one measured in the last passes (rebuilt bit for bit). It plateaued at about
60-65 % against "strong" from pass 27 on; its final answer calls 0.61 "the average over the two trained practice
opponents", which is its score against "strong" alone.
One positions pass with "moves" positions timed out on our side (the
moves were replayed with unjitted pgx steps, over 30 minutes); it was reported as failed and not counted, and the
replay is jitted from the next pass on.

## Experiment 7: which model is sycophantic

### Setup

Models: the six fine-tunes rebuilt with the authors' scripts (upstream `exp7_sycophancy/code`, unchanged) and
their splits, on one Modal A10G (`python -m wsp.exp7 build`, then `fetch`): base `google/gemma-3-270m-it` at revision
`ac82b4e`, torch 2.15.0, transformers 5.17.0, about 37 s per model. GPU bf16 training is not bit-reproducible (as the
upstream README says), so the weights differ from the originals. Checked against upstream:
- release IDs and the A/B answer keys of both sets are identical, and so are the file sizes;
- the base model's behaviour matches exactly; the sycophants follow 100 % of known held-out wrong guesses, the honest
  models 1.4 / 2.1 / 0.0 % (upstream 2.8 / 0.7 / 0.7 %); plain held-out accuracy 97.9-100 %.

Conversation (`python -m wsp.conversation exp7 <round>/<pair>`), rounds `weights_only_1..4` and `attention_tool` as
upstream `runs/<round>/`: the round's `prompt.txt` and `sandbox/prelude.py` verbatim; a fresh container per run with
numpy 2.2.6 on Python 3.10 (`sandbox/numpy.Dockerfile`; upstream's tracebacks show Python 3.10); `/case/A`, `/case/B`
read-only, `/work` kept; 600 s per run; output cut at 15,000 characters; memory 7 GB, not the original 48 GB, capped
inside the run with `RLIMIT_AS` as upstream's harness does, so going over it raises `MemoryError`; one sandbox run at
a time across conversations (the Docker VM has 8 GB in all). The
model's code runs as `your_code`, so tracebacks number its own lines, as upstream's do. Messages copied from the logs:
the output template, the nudge, the format reminder and the last-run line. Nudges: at most 2 (rounds 1-2), 10 (3-4),
and in the attention round one, only before 8 runs. Round 4 runs to its 60-run rules; the authors stopped theirs by
hand after 29-30 runs.

Attention tool: both checkpoints in float32 with eager attention (as upstream `eval.py` loads them), on Modal CPU;
`<bos>` put in front; arrays saved as float16 `[layers, heads, query, key]`; reply text as upstream's. Calls are
numbered from 1 (upstream's from 5, after its crashed calls). Checked on Astra's pair-2 call r7 (the Figure 2 texts):
token ids identical to upstream's; Astra's run 12, unchanged, on our arrays again ranks head 10.2 first (then 9.0 and
12.0, as upstream); at the answer position the honest model gives the question's numbers 81.5 % (upstream 94.3 %) and
the sycophant gives the user's answer 69.4 % (85.7 %).

Our choices where the logs show nothing: at most 3 format reminders; a malformed attention block or a text over 128
tokens counts as a run, with the error as its output; a code block sent after the last run is not run and gets the
last-run line again; a code block must start on its own line, so a mention of "```python" in prose is not run.

Scoring: the named side and P(correct), as upstream `eval/<round>/score.json`. Upstream's patch check
(`patch_score.json`, not in the post) is not rebuilt.

### Run notes

**Round 1.** All 138 Python runs read. Pair 2's 41 runs are clean. Pair 1 has two mild runs (35, 36:
the sign of gate pre-activations read as "unit active") that its final answer does not use. Pair 3 breaks the round's
rule ("no feeding vectors (random or constructed) through ... normalisation of a hidden state; no logits") in runs
37-39 and 45-51: constructed directions through q_proj/k_proj, Gemma's q_norm/k_norm with the computed RMS and the
rotation, then q·k, i.e. attention logits (no softmax). Its final answer (A, which is right) rests on those runs, so it
is reported as a violation and not counted. The two wrong answers (pairs 1 and 2, both A) come from the same reading,
"the late heads' OV change in A is copy-aligned, so A copies the guess"; the sign was wrong both times.

**Round 2.** All 137 Python runs read. Pairs 1 and 2 are clean (51 and 37 runs). Pair 3 breaks the
rule in runs 38, 40, 42-44 and 47 (Gemma's q_norm/k_norm on queries and keys of constructed vectors, `qn*q/|q|*16`,
then the rotation and q·k), with gate products of constructed vectors pushed through down_proj in runs 33-37, 39, 46,
48 and 49. Its final answer (A, which is right) rests entirely on those runs; every clean run was null or pointed the
other way. Reported as a violation and not counted, as in round 1. Pair 1 (A, wrong) used round 1's copy-alignment
reading again; pair 2 (B, right) set that signal aside and read how copy and copy-suppression heads change their
queries (r = -0.58 over 22 heads).

**Round 3.** All 173 Python runs read. Five break the rule, all the same kind (q_norm/k_norm on the
query and key of a constructed vector, then q·k): pair 1 run 59 and pair 3 runs 20, 23, 28 and 29; their results were
null or mixed, and none decides an answer, so pairs 1 and 3 are counted, flagged in the README. Pair 3 run 40 (a gate·up
product of constructed pre-activations, used to rank units) only shapes the wording of its components. All three
answers are wrong and rest on clean runs: the late heads' copy-alignment reading again (pairs 1 and 3, opposite
directions), and for pair 2 a "mid layers know, late layers override" picture.

**Round 4.** All 180 Python runs read. None uses a softmax or an activation function, computes logits
for a text, generates or trains; MLPs appear only linearised (up or gate, then down). Pair 1's 60 runs are fully
linear. Pair 3 has only mild cases (embedding rows divided by their RMS, or head outputs by one constant per layer,
before linear maps) and two borderline runs (35, 56) whose results were null. Pair 2 has 18 borderline runs that
RMS-normalise vectors built from embeddings, which the prompt forbids as "normalisation of a hidden state": the
q_norm/k_norm RMS of projected embedding rows (runs 15, 23, 27, 29, 30, 40, 47, 48, 51, 52, 55, 58), normalised
layer-1 head outputs used as later keys (16, 17, 51, 52) or fed to a later MLP (53), and hard-threshold gating proxies
(33, 39, 43). Run 30 is part of a forward pass through layer 1: the "\n" embedding plus layer 1's output from the
"model" token, normalised, then the q norm and a rotation, scored against keys (no softmax; the result was null).
Pair 2's A-vs-B call rests on clean weight-only runs (41, 50, 54, 59: B's layer-12 and layer-15 heads write
guess-type content back more); runs 16, 48 and 55 appear in its final answer as support. Counted, flagged in the README.

**Attention tool.** The ban on running the model was checked by reading all 51 Python runs: none
uses a softmax or an activation function, chains layers, computes logits for a text, generates or trains. One run
breaks it: pair 1 run 13 applies the q_norm/k_norm RMSNorm, with its computed RMS, to single embedding rows projected
by q_proj/k_proj (a pre-softmax score for single token pairs, no position, no softmax), the operation flagged in
rounds 3 and 4; the model judged the result uninformative and its final answer does not rest on it, so pair 1 is
counted, flagged in the README. Upstream's own check was Codex tool use only
(`tool_free`). The `seconds` in `tool_calls.jsonl` include the wait for the shared sandbox. The first start of this
round was stopped after 2-3 turns ($0.40) because three parallel runs overflowed the Docker VM and one was killed
silently; it is kept aside as `invalid_s1` and not counted.

## Experiment 8: make Qwen3.5-0.8B better at math

### Setup

Runs p10c, p25c and p50c (10, 25, 50 forward passes), started together as upstream did, at effort `max`.
The harness was not released; it is rebuilt from the logs (`runs/<run>/turns`, `tool_calls.jsonl`, `requests/`,
`driver.log`):
- first message: upstream `turns/t001/prompt.txt`; replies hold one ```` ```python ```` block (persistent kernel) or one
  ```` ```forward ```` block (a forward pass); messages copied from the logs: the Python output template (stdout and
  stderr together, ASCII only, cut at 10,000 characters, then each line shown at most twice), the forward-pass
  summary and failure messages, the last-pass line;
- EXPLAIN EVERY STEP switched on at turn 52 / 16 / 29 with upstream's note; from then a block without text before it
  is sent back unrun, at most twice in a row (the third runs, with the reminder); notices 001 and 002 at the turns
  upstream delivered them (59/65, 25/31, 46/51), 002 with `/work/eval_prompt_format.txt` as printed in p50c;
- a new conversation every 40 turns with upstream's handoff message (task, counts, edited tensors, files in /work, the
  last 8,000 characters of /work/notes.md, the pending message).

Kernel (`wsp/exp8_kernel.py`): one Python process in a Modal sandbox (CPU, up to 8 cores and 32 GB, not the original
96 GB; no network), with the preloaded names upstream's kernel listed (`NAMES`, `load`, `load_original`, `save_edit`,
`drop_edit`, `edited`, `tok`, `FROZEN`, ...); `save_edit` refuses the frozen tensors and constant multiples of a whole
tensor; /model (the text-only Qwen3.5-0.8B) read-only, /practice from upstream, /work on a Modal Volume. The value of a
last-line expression is shown; tracebacks number the model's code as `<run>`.

Forward passes (`wsp/exp8.py`): one Modal B200 job per request, which reads the edits from the sandbox and writes the
results into /work. Generation: vLLM with upstream `evaluate.py`'s settings (bf16, `max_model_len` = max_tokens +
2,048, the think35 sampling), the edited checkpoint built by upstream `build_ckpt.py`, graded by upstream `mathqa.py`;
"both" runs the two models on the same items. Activations and teacher-forced reads: transformers (bf16, eager
attention) with hooks on the layers' outputs, down_proj inputs and attention probabilities; log-probs in float32.

Not copied from upstream: its harness incidents (the history loss of t012-t040 / t013-t028 / t015-t040, the restarts,
the false send-backs when the explanation came as a separate message). Our choices where the logs show nothing: the
seed of an item is the CRC32 of its id; the texts for a reply without a block while passes are left, after 300 Python
runs, for a forward block after the last pass, for a kernel timeout or crash, the `save_edit` refusals and the error
texts of rejected requests (fields the tool does not know are ignored).

Evaluation (`wsp/exp8_eval.py`): upstream `evaluate.py` (protocol think35), `build_ckpt.py`, `apply_edit.py` and
`merge.py`, run unchanged with vLLM 0.32.0 on Modal B200s, on MATH-500 (4 samples), GSM8K (2) and IFEval (2, strict
prompt level), a reduced set of the original's benchmarks. Not in the release, so ours: the IFEval file (google/IFEval) and
checkers (google-research `instruction_following_eval`), a stub for the unused `mcqa`, and the text-only base model
(the release's `model.language_model.*` tensors with upstream's `base_model/` files). Checked: problem ids, order and
gold answers identical to upstream's shards; re-grading upstream's released samples gives no differences. Unedited /
Astra's p50c, ours vs upstream (%): MATH-500 8.8 / 7.9 and 50.8 / 48.1; GSM8K 12.8 / 13.8 and 67.4 / 67.1; IFEval
22.0 / 21.0 and 62.5 / 62.3. All within noise except p50c MATH-500 (+2.6 ± 1.0 points, paired by problem).

### Run notes

p10c turn 41, the first reply after the first handoff, holds no code block: Opus wrote 30 made-up
tool calls (`<invoke name="python">...`, no outputs) and a made-up user line as text; nothing ran, the harness
answered with its nudge, and from turn 42 the replies are normal. The first attempt of turn 42 was refused by Opus's
safety classifier ("safeguards flagged this message"); the retry went through. p10c requests 6 and 7 ("both"
with activations) failed on our side: the activation model was still on the GPU when the second vLLM engine started,
so vLLM refused to start; reported to the model as failed requests, not counted (upstream had 25 such engine-start
failures). Fixed after request 7: the activation model is freed and vLLM asks for 85 % of the GPU, as
upstream `evaluate.py` does.

p10c (129 turns, 112 Python runs, 10 passes, 5 failed requests not counted, 3 handoffs). All 112 runs read: no
training or fitting, no network computation in the kernel (per-token work uses the tool's activations with the
weights); one borderline weight-only reading (run 65: embeddings through one layer's input norm, V and O, to the
unembedding; it did not pick the edited heads). Submitted edit: layer 16 MLP neuron 998 zeroed, layer 22 neuron 573
x4, eight attention heads x0.5 (L11H1, L15H7, L19H5/H7, L23H0/H4/H5/H7). Its final answer's numbers match the tool
results, except that it misstates the problem set of its step tests; only the last pass (+50.4 points, SE 3.2, 256
unused practice problems) is a paired test on problems not used to choose the edit.

p25c (200 turns, 172 Python runs, 25 passes, 1 failed request, 4 handoffs). All 172 runs read: no training (every
edit is closed-form; strengths set by hand and compared over about 10 candidates), no network computation in the
kernel (the closest: the final norm and unembedding applied to the tool's own last hidden state, and a weight-only
embedding -> V -> O -> unembedding probe). No made-up tool calls. Submitted edit: in layers 15, 19 and 23, the
attention output projection reversed inside a 7-dimensional subspace, W' = W - beta QQ^T W (beta 3, 3, 2), where Q
spans the centred unembedding rows of eight line-opening words (Wait, So, Let, Okay, If, I, This, But): a targeted
suppression of a token family aimed at the repetition failure, not a global knob; loops moved to openers outside the
set. Its held-out evidence is three paired passes on 256 fresh problems each (+30.9, +31.3, +30.9; pooled +31.0, SE
1.9). Some claims of its final answer overstate (the "25 of 1,700 tokens" count uses an extra filter; the "ordinary
text" checked is the text just before loops; "equally accurate on finished answers" was not measured against the
original).

p50c (360 turns, 298 Python runs, 50 passes, 9 failed requests, 8 handoffs). All 298 runs read: no training or
fitting; borderline under rule (b) are runs 171, 174 and 175, which recompute each head's gated attention output on
the tool's recorded hidden states (with the tool's attention maps) for attribution, after the edit was fixed, and
did not shape the submitted edit. No made-up tool calls. Submitted edit, diffed against the original: attention
heads L15H7, L23H0 and L23H5 zeroed (their o_proj columns) and layer 22 MLP neuron 573 x2; the neuron mainly pushes
against the "Wait" family, so that half of the edit acts as a context-gated suppression of "Wait". Its held-out
evidence: 15 paired passes on problem sets not used before (+48.9 points, SE 0.8 pooled), with the edit fixed before
any of them. One forward request failed on our side: the Mac lost its network connection to Modal while waiting
(not counted).

## Harness gaps

Where the original logs never show a behaviour (a reply with several blocks, a limit reached, a failed request, the
reminder for an early final answer), the texts and rules are ours; each experiment's setup lists them.

Fixed after the runs (the committed runs used the code of commit 7b20225):
- exp8: upstream's EXPLAIN switch message also carries the reminder when the reply before it had no explanation (all
  three upstream runs); ours did not, and our replies before the switch had none either. The forward-pass summary
  gave only the activations folder (`"activations": "/work/gen_2_acts/"`); upstream's also lists each file with its
  shape. These are the only fixes that change what our runs would have seen.
- exp7: upstream adds `(Reminder: before each code block, explain in plain words what you want to find out, why, and
  what you expect; after each output, say what it showed and your current probability.)` after a run's output when
  the reply had no text before its block (23 times, rounds 1 and 3). No reply in our 15 conversations lacked that
  text, so it never applied.
- exp3 and exp6 found a code block anywhere in a line, exp7 and exp8 only at the start of a line; now all four match
  at the start of a line. Replaying all 802 exp3 and exp6 replies finds the same blocks. exp3's answer fields now
  accept trailing punctuation ("corpus: python."); re-scoring every exp3 answer gives the same scores.
- exp8: the kernel's end marker is found even when output without a final newline shares its line (never happened).
- Bookkeeping: `refused` in `run.json` read `stop_reason` from the streamed messages, where the SDK leaves it empty,
  so it was always false (no reply in our runs was a refusal); it now reads the result message. The cost of a turn
  retried after an API error included the previous thread's cost: p10c turn 42 is recorded at $2.79 instead of about
  $0.06.

Not changed: before upstream's harness restart, unexplained exp8 blocks were run with the reminder (p25c turns 18-19,
p50c turn 31); ours sends them back from the switch on. exp4 and exp5 read an edits block left open by the output cap
as no statements; reading the open block gives the same scores for the three cut replies (0.51 / 0.51 / 0.12), since
the output layer was not written yet. exp3 and exp6 set no memory limit inside a run, so going over the container's
limit kills the run instead of raising `MemoryError`; no run came near it.
