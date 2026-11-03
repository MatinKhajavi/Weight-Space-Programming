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

Each run is stored in `runs/<model>/<exp>/<item>/s<sample>/`.

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
   `answer_cut.md` and its score is recorded in `score.json` under `cut_at_output_cap`.

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
- Each environment and evaluator is checked against Astra's released artifacts before any Opus run, for example the
  exp4 base network at 0/256 and the exp5 networks at about 517 / 53 / 53.
- Where the original seeds are not recorded (exps 4, 5), comparisons are statistical, not exact.

### Validation so far

- **exp1, exp2 scorers** (`wsp/scoring.py`): on Astra's 7 released answers they give the same mapped answer, the same
  right/wrong and the same confidence as the released `eval/scores.json` (4/4 games, 3/3 shapes). Our keyword rules:
  names only (a description without a name is wrong); two different labels named is wrong ("hedged"); whole-word
  matching; exp2 scores the shape only, not filled vs outline.
- **exp5 edit reader** (`wsp/exp5.py`): rebuilds Astra's three released networks bit for bit, no statement skipped.
- **exp5 evaluator**: official MinAtar 1.0.15 `Environment`, one environment per worker with episodes back to back
  (its last action carries over between episodes, as in the official wrapper), fixed seeds. Astra's Breakout network
  scores 506.5 ± 7.6 (seeds 0-999) and 508.1 ± 7.6 (seeds 1000-1999) against the released 517.4 ± 7.4: about 1
  standard error low in both blocks. Astra's and Opus's networks are scored with the same evaluator and seeds, so a
  small offset between our evaluator and theirs does not affect the comparison.

## Order of work

1. exp1 Breakout and exp5 Breakout first, to confirm the setup end to end before the rest.
2. The rest of exps 1, 2 and 5, one sample each.
3. The exp4 model call; scored once the Kinetix evaluation reproduces the original base result.
4. The multi-turn experiments (3, 6, 7, 8). Their harness gaps (listed below) get designed and written down here
   before any run.

## Open: multi-turn harness gaps

Never observed in the original logs, so the original behaviour is unknown:

- exp6: the reminder text sent when the model answers with passes left.
- all: handling of a reply with several code blocks, or prose next to a block.
- all: behaviour when a run or pass limit is reached, and the message for a failed request.
- exp3: the sandbox prelude that preloads the weights as `W` (its code is not released; described in `message.txt`).
