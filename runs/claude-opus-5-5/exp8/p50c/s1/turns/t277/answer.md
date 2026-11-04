A stronger brake has nothing left to act on. "Actually" (0.12–0.15 per 1,000 tokens) and "Hmm" (0) are rare and no more frequent in failures than in successes. The words that are enriched in cut texts ("So" 7.2 vs 3.4, "Maybe" 3.4 vs 1.0, "This" 2.9 vs 1.6) are mostly inflated by the loop repetitions themselves, and suppressing "Maybe" already failed. So I skip the ×3 dose: the brake route is saturated.

One copying head from the original loop analysis has never been tested on its own: 23.7. It had 0.47 in-loop vs 0.18 pre-loop induction mass in the original model and is still copying in C loops (0.43). It sits in the same layer as the two induction heads whose removal helped most. Its only test was as part of edit B, together with 19.5, 19.7 and 23.4. B was harmful, but 23.4 alone has since proven neutral, so the damage came from some combination of 23.7, 19.5 and 19.7. I will test X4 = C + 23.7 zeroed on the same 512 items. A paired gain clearly above about 2 points would merit held-out confirmation. A loss would confirm 23.7 as load-bearing and end head ablation for good.

```python
# Set X4 = C + head 23.7 zeroed (layer-23 o_proj cols 7*256:8*256, on top of 23.0/23.5 zeroed); verify
W23 = load('model.language_model.layers.23.self_attn.o_proj.weight').clone()
W23[:, 7*256:8*256] = 0
save_edit('model.language_model.layers.23.self_attn.o_proj.weight', W23)
W0 = load_original('model.language_model.layers.23.self_attn.o_proj.weight')
print('L23', [round(float(load('model.language_model.layers.23.self_attn.o_proj.weight')[:,h*256:(h+1)*256].norm()/W0[:,h*256:(h+1)*256].norm()),2) for h in range(8)], edited())
with open('/work/notes.md','a') as f:
    f.write("- Session 4 checks: brake col z on closing tokens negligible (Final +.7, ** 0, Done -.8); commit loops 5/82. 'Final' neurons (23.1238 z+24, 17.838 +20, 19.734 -18) fire on prompt 'put your final answer', silent in loops -> no latent conclude impulse. Actually/Hmm line starts rare (0.1/1k) -> brake x3 pointless.\n- X4 = C + head 23.7 zeroed. Pass 21 (planned): edited, SH512.\n")
```