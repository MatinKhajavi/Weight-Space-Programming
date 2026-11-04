The copy-suppression heads are not engaged in loops: 23.2 puts 0.00 attention on earlier copies of the about-to-be-predicted token, and 11.0, 11.6 and 15.3 put 0.05-0.19. So there is no natural anti-repetition head to amplify. Head 19.5 is the stand-out. At the decision positions of loop lines it puts 0.74 of its attention on earlier copies of the next token, against 0.10 on pre-loop lines, a sevenfold increase. It is the most loop-specific remaining head; 23.7 is a distant second (0.33 vs 0.10). As a cheap causal filter before spending a generation pass, I'll zero only 19.5's o_proj columns on top of the current edit. Then I'll rerun the same two windows teacher-forced (logprobs, hidden states, MLP activations) and compare the copy-token probabilities at the decision points with fwd_18. If 19.5 drives the lock-in, p(copy) on the early repeats (0.43 → 0.88 now) should fall clearly. A negligible change would refute it and save a generation pass.

```python
# Hypothesis test setup: C + head 19.5 o_proj columns zeroed (single-head ablation, other edits unchanged)
import shutil, os
nm19='model.language_model.layers.19.self_attn.o_proj.weight'
W=load_original(nm19).clone(); W[:,5*256:6*256]=0.0
save_edit(nm19, W)
print(edited())
print('19.5 cols zero:', float(load(nm19)[:,1280:1536].abs().sum()), 'other cols unchanged:', float((load(nm19)[:, :1280]-load_original(nm19)[:, :1280]).abs().sum()))
```