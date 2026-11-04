Half-strength heads (V-b) score 60.0% against C's 60.7%, essentially equal. But the outcome mix shifts: cut answers rise from 24.8% to 32.0% while wrong answers fall from 75 to 41. Partly restoring the line-start copying re-creates loops, and these occur mostly on problems the model would otherwise answer wrongly. This is the same cut-versus-wrong trade-off seen with the 23.4 ablation. It confirms that the remaining loops mostly absorb problems the model cannot solve, so pushing the copy heads further (they are already at zero) has no accuracy to gain. Both dose curves now place C at the optimum: brake ×2 (×2.5 is worse), heads at 0 (equal accuracy to 0.5, with fewer loops). Next, the head factorial under the final brake: brake ×2 with only 15.7 zeroed, then with only 23.0/23.5 zeroed. This shows whether both copy routes (layers 15 and 23) are still needed once Wait is braked. Without the brake each route contributed about half of the head effect (+10.2 and +14.1 vs +22.7 together).

```python
# Record V-b; set factorial F15 = brake x2 + only 15.7 zeroed (23.0/23.5 restored)
VB=smap(['/work/gen_36.jsonl']); rb=pdiff(VB,C)
print('V-b vs C', rb)
with open('/work/notes.md','a') as f:
    f.write(f"- PASS 30 = gen_36: V-b = copy heads x0.5 + brake x2 on SH512: acc 60.0 (gsm 71.1, math 48.8), cut 32.0 (C 24.8), wrong 41 (C 75); paired vs C {rb[1]} SE {rb[2]}. Head dose under brake: 0 ~= 0.5 in acc, 0 has fewer loops (cut<->wrong trade). C keeps 0.\n- Pass 31 (planned): F15 = brake x2 + only 15.7 zeroed, SH512.\n")
Wc=load_original('model.language_model.layers.15.self_attn.o_proj.weight').clone(); Wc[:,7*256:8*256]=0; save_edit('model.language_model.layers.15.self_attn.o_proj.weight',Wc)
drop_edit('model.language_model.layers.23.self_attn.o_proj.weight')
print(edited(), 'L23 original:', bool(torch.equal(load('model.language_model.layers.23.self_attn.o_proj.weight'),load_original('model.language_model.layers.23.self_attn.o_proj.weight'))), 'brake x2:', bool(torch.equal(load('model.language_model.layers.22.mlp.down_proj.weight')[:,573],2*W22o[:,573])))
```