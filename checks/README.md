# Checks

Scripts behind the validation and run notes in [docs/protocol.md](../docs/protocol.md). Run them from the repo root,
for example `uv run python checks/exp5_validate.py`. Scripts that use Modal need `MODAL_PROFILE` in `.env`.

| What it checks | Script |
|---|---|
| Nothing from the local setup reaches the model call (no CLAUDE.md, memory, tools) | `capture.py`, `capture_server.py` |
| The multi-turn loop and its retry rewind, on scripted replies | `conv_check.py`, `rewind_check.py`, `exp6_conv_check.py` |
| exp4 evaluator: the network as shown and Astra's edit give the released results | `exp4_modal.py` |
| exp5 evaluator: Astra's networks give the released scores | `exp5_validate.py` |
| exp5: how a written Breakout network loses the ball | `exp5_breakout_losses.py` |
| exp6 Go engine: Astra's networks against the released PPO networks give the published results | `exp6_modal.py` (result in `exp6_validation.json`) |
| exp6 rebuilt opponents | `exp6_opp_verify.py` |
| exp6 budget 0: who makes the repeating move | `exp6_rep.py`, then `exp6_cycle.py` |
| exp6, exp7 and exp8 turn handlers on scripted replies | `exp6_step_test.py`, `exp7_step_test.py`, `exp8_step_test.py` |
| exp7 rebuilt models: behaviour, release IDs and answer keys | `exp7_behaviour.py` |
| exp7 attention tool on the Figure 2 texts | `exp7_fig2.py` |
| exp8 kernel and forward-pass tool | `exp8_harness_test.py`, `exp8_forward_test.py` |
| exp8 evaluation against upstream's | `exp8_regrade.py`, `exp8_compare.py` |
