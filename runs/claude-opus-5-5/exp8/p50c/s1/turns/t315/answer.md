The brake at ×2.5 is slightly worse than ×2: 58.2% against 60.7% on the same seeds. Cut answers barely move (23.4% vs 24.8%), but wrong answers rise (94 vs 75). Wait is already essentially gone at ×2, so the extra strength mostly increases the off-target suppression of other self-correction words and closing tokens ("Hmm", "Actually", "Ready", </think>), which seems to cost correctness. The full dose curve is now ×1 35.2 → ×1.5 58.2 → ×2 60.2/60.7 → ×2.5 58.2: ×2 sits at the optimum, and the rule "back off if it hurts" confirms it. Next I test the other axis: the three copy heads at half strength (o_proj blocks ×0.5) instead of zero, with the brake at ×2. If half strength matches C, a gentler edit would do. If it is worse, it shows the copying must be removed fully. Without the brake, the head curve was 1.0 → 0 → -0.5 = +0, +22.7, +13.7 points.

```python
# Record V-a (brake x2.5); set V-b = heads 15.7/23.0/23.5 at x0.5 + brake x2
VA=smap(['/work/gen_35.jsonl']); ra=pdiff(VA,C)
print('V-a vs C', ra)
with open('/work/notes.md','a') as f:
    f.write(f"- PASS 29 = gen_35: V-a = C with brake x2.5 on SH512: acc 58.2 (gsm 68.0, math 48.4), cut 23.4, wrong 94 (C 75); paired vs C {ra[1]} SE {ra[2]}. Brake dose: x1 35.2 / x1.5 58.2 / x2 60.2-60.7 / x2.5 58.2 => x2 optimal.\n- Pass 30 (planned): V-b = copy heads at x0.5 (not 0) + brake x2, SH512.\n")
W22=W22o.clone(); W22[:,573]=2*W22o[:,573]; save_edit('model.language_model.layers.22.mlp.down_proj.weight',W22)
for l,hs in ((15,[7]),(23,[0,5])):
    Wc=load_original(f'model.language_model.layers.{l}.self_attn.o_proj.weight').clone()
    for h in hs: Wc[:,h*256:(h+1)*256]*=0.5
    save_edit(f'model.language_model.layers.{l}.self_attn.o_proj.weight',Wc)
w23=load('model.language_model.layers.23.self_attn.o_proj.weight'); o23=load_original('model.language_model.layers.23.self_attn.o_proj.weight')
print('V-b set:', bool(torch.equal(w23[:,0:256],0.5*o23[:,0:256])), bool(torch.equal(load('model.language_model.layers.22.mlp.down_proj.weight')[:,573],2*W22o[:,573])))
```