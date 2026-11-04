The onset is not a hesitant decision. In window 0, the first copy is already near-deterministic (mean log-probability −0.014, worst token −0.21, an indentation token). In window 1, the first copy has a few uncertain tokens (down to −1.4) but locks by the second cycle (−0.011). So the model enters an exact copy of the earlier segment with high confidence at the first re-entry. That comes from a fuzzy context match, "same kind of line start, so same continuation", which then becomes exact token copying. It is attention-based retrieval of the earlier segment, consistent with the earlier finding that ablating induction heads was the main lever.

Among heads that retrieve earlier line starts, only one strong candidate has never been tested: 11.1, the top in-loop copier under C (attention mass 0.67 on previous-cycle targets). I will test it directly as X3 = C + head 11.1 zeroed, on the same 512 S1+H1 items, for a paired comparison with C. As in the X1 cross-tab, the transitions will tell whether any loop rescue is offset by damage to finished reasoning. A paired gain clearly above its standard error (about 2 points) would make it worth confirming on fresh held-out problems.

```python
# Set X3 = C + head 11.1 zeroed (o_proj columns 256:512 of layer 11); verify edits
W11 = load_original('model.language_model.layers.11.self_attn.o_proj.weight').clone()
W11[:, 1*256:2*256] = 0
save_edit('model.language_model.layers.11.self_attn.o_proj.weight', W11)
for L in (11,15,23):
    W = load(f'model.language_model.layers.{L}.self_attn.o_proj.weight'); W0 = load_original(f'model.language_model.layers.{L}.self_attn.o_proj.weight')
    print(f'L{L}', [round(float(W[:,h*256:(h+1)*256].norm()/W0[:,h*256:(h+1)*256].norm()),2) for h in range(8)])
print(edited())
SH = json.load(open('/work/SH512_ids.json')); print(len(SH), SH[:2])
with open('/work/notes.md','a') as f:
    f.write("- Session 4: C loop windows (fwd_18) neuron DLA at line starts: no loop-specific brake (23.932 -1.2 but same at control); loop excess +2.2 logits spread (L23 MLP +1.1, L22 MLP +.55, L20 mixer +.35). Onset: first copy already near-deterministic (cycle1 mean lp -0.014/-0.115) -> loops entered by fuzzy line-start retrieval then exact copying.\n- X3 = C + head 11.1 zeroed. Pass 20 (planned): edited, SH512, compare with C (gen_13+gen_16_edited).\n")
```