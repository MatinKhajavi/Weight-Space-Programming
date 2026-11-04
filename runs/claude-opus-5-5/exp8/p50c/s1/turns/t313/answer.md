Language or rare-token drift is not a factor. Chinese/Japanese/Korean characters appear in only 2–3% of texts (3.4% of cut, 3.2% of wrong, 1.6% of right answers), and other unusual non-ASCII characters are essentially absent. So the presence penalty is not pushing C into degenerate tokens, and the remaining failures stay in the "stuck on hard problems" category. C is my final edit. With the remaining passes I will first map the dose-response around C's operating point on SH512, against C's 60.7% on the same seeds. This shows whether C sits at, or near, the optimum of both its components. Up to ×2, the Wait-brake curve rose steeply and then flattened (×1 35.2, ×1.5 58.2, ×2 60.2 on S1). So first ×2.5 with the heads still zeroed: a further gain would mean the brake was under-dosed, and a loss would mark the upper side of the optimum. After that I will try the heads at half strength, then run fresh confirmation sets.

```python
# Dose-response around C: set V-a = C with Wait brake x2.5 (heads 15.7/23.0/23.5 still zeroed); log drift check
with open('/work/notes.md','a') as f:
    f.write("- Drift check (C, 1280 answers): CJK in 1.6% ok / 3.2% wrong / 3.4% cut texts, odd non-ASCII negligible => not a failure mode.\n- Per-source paired diffs of C-variants on SH512: none positive on MATH (X1 +0.8, X3 -6.6, X4 -5.5, X6 -3.9, X7 -2.0).\n- Pass 29 (planned): V-a = C with brake x2.5, SH512.\n")
W22=W22o.clone(); W22[:,573]=2.5*W22o[:,573]; save_edit('model.language_model.layers.22.mlp.down_proj.weight',W22)
w=load('model.language_model.layers.22.mlp.down_proj.weight')
print(edited(), 'brake x2.5:', bool(torch.equal(w[:,573],2.5*W22o[:,573])), 'heads zero:', bool(load('model.language_model.layers.23.self_attn.o_proj.weight')[:,0:256].abs().sum()==0))
```