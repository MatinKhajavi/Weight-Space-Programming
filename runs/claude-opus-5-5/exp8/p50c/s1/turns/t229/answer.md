Prompt-directed induction is tiny for every head (0.006–0.032) in these short pre-loop stretches, so this measure cannot separate useful copying from loop copying. The decisive test has to be behavioural. Among heads still active under C, the strongest loop copiers are 11.1 (loop induction 0.67) and 23.4 (0.53). 23.4 also does 0.86 of the line-start retrieval that begins each loop cycle and writes directly into the logits. Edit B removed 23.4 together with 23.7, 19.5 and 19.7, before the brake existed, when loops were 'Wait'-driven, so its separate role under C is untested. I will test C plus zeroing head 23.4 on all 512 already-measured items (S1 and H1). This gives a paired, same-seed comparison with C's existing outputs (gen_13, gen_16) at about twice the power of a 256-item pass. Fewer cut-offs without more wrong answers would confirm 23.4 as a loop driver; more wrong answers would mean it is needed for computation, and I would revert.

```python
# Build X1 = C + head 23.4 zeroed (o_proj cols 1024:1280), keep C copy for revert; write problem list for a 512-item paired pass
W23 = load('model.language_model.layers.23.self_attn.o_proj.weight').clone()
print('23.0/23.5 zero already:', float(W23[:, 0:256].abs().sum()), float(W23[:, 1280:1536].abs().sum()), '| 23.4 norm before', round(float(W23[:, 1024:1280].norm()), 2))
W23[:, 1024:1280] = 0
save_edit('model.language_model.layers.23.self_attn.o_proj.weight', W23)
print(edited())
ids512 = S1 + H1; print(len(ids512), len(set(ids512)))
json.dump(ids512, open('/work/SH512_ids.json', 'w'))
with open('/work/notes.md', 'a') as f:
    f.write("\n- gen_21 (pass 16, undocumented at the time): variant on S1 with line-initial Maybe 2.67->1.70/1k, Wait same: acc 60.5% vs C 60.2% (W->R 18, R->W 17), cut 27.3% => no gain; reverted to C.\n"
            "- Session 3 analyses: C failures by source: GSM ok70/cut14/wrong16%, MATH ok52/cut35/wrong13% (L1-5 ok 86/78/52/42/31). Strict commit failure only 25/126 cuts (5% of answers). GSM wrongs: genuine misreadings, ref computed before wrong only 5/41.\n"
            "- </think> under C in loop windows: lp median -9..-11 -> brake's </think> suppression irrelevant. Indentation: deep nesting correlates with failure (GSM fail 56% >=24sp @100 lines vs ok 15%) but indent choice is ~deterministic (margin 8-12) -> presence penalty not flipping; symptom.\n"
            "- Head 23.2 OV strongly suppressive but attends nowhere relevant (ind .02) -> not a loop brake. Loop copying under C: 11.1 (.67), 23.4 (.53, line-start retrieval .86), 19.5 (.46), 23.7 (.43).\n"
            "- X1 = C + 23.4 zeroed. Pass 17 (planned): edited, S1+H1 (512, /work/SH512_ids.json), compare with gen_13 + gen_16_edited.\n")
```