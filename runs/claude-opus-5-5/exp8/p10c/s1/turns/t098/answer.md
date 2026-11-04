Neuron 573 in layer 22 is a clean negative-feedback brake on doubt:

- **Output:** its down-projection column points toward the Wait tokens (' wait', 'Wait' and ' Wait' are among its top-promoted tokens), but its activation is strongly negative exactly at doubt decisions (−4.3/−3.6, against about −1.5 at other line starts). So in practice it suppresses "Wait".
- **Input:** it reads the "Wait is predicted" direction (gate +0.52, up −0.42), including the output of the ablated neuron 998.

So the more the residual stream predicts Wait, the harder it pushes back. Amplifying it is a principled, conditional anti-doubt edit for later, with low risk because it only acts where Wait is already being predicted.

The main failure, though, is exact repetition (144 of 152 cut answers), and the evidence says the lock-in is written mostly by late full-attention copy heads that v2 left untouched. I will therefore test that first, alone, for clean attribution.

Edit v3 is v2 plus halving the output of layer 23 heads 5, 0, 4 and 7 and layer 19 heads 5 and 7. These had the highest in-loop attention to the previous cycle (0.36–0.74) but weaker induction before the loop (0.13–0.24) than the heads already halved. The test pass will cover setA (paired with v2's identical-seed texts) plus 256 diagnostic problems (paired with the original's pass-1 texts).

If v3 reduces exact loops and raises accuracy over v2, the copy-lock-in account is confirmed. If loops stay and accuracy drops, these heads are needed for normal copying and I will back off to the neuron 573 route. Now I build and verify the edit and draw the 256 diagnostic problems.

```python
# Hypothesis: (setup) build v3 = v2 + late copy heads L23 H0/H4/H5/H7 and L19 H5/H7 output x0.5; verify only those columns change; draw diagS (192 MATH + 64 GSM8K from diag set)
HD = 256
def head_scale(L, heads, f):
    name = f'model.language_model.layers.{L}.self_attn.o_proj.weight'
    W = load(name).clone()
    for h in heads: W[:, h*HD:(h+1)*HD] *= f
    save_edit(name, W)
head_scale(23, [0, 4, 5, 7], 0.5); head_scale(19, [5, 7], 0.5)
for L in [11, 15, 19, 23]:
    name = f'model.language_model.layers.{L}.self_attn.o_proj.weight'
    W, W0 = load(name), load_original(name)
    ratio = [(W[:, h*HD:(h+1)*HD].norm() / W0[:, h*HD:(h+1)*HD].norm()).item() for h in range(8)]
    print(L, W.shape, 'head norm ratios', np.round(ratio, 3))
d16 = load('model.language_model.layers.16.mlp.down_proj.weight'); d16o = load_original('model.language_model.layers.16.mlp.down_proj.weight')
print('L16 changed cols:', torch.nonzero((d16 - d16o).abs().sum(0) > 0).flatten().tolist(), '| edited:', edited())
diag = json.load(open('/work/diag_ids.json')); setA = json.load(open('/work/setA_ids.json'))
rng = np.random.default_rng(77)
dm = [x for x in diag if x.startswith('math')]; dg = [x for x in diag if x.startswith('gsm')]
diagS = list(rng.choice(dm, 192, replace=False)) + list(rng.choice(dg, 64, replace=False))
assert not set(diagS) & set(setA)
json.dump(diagS, open('/work/diagS_ids.json', 'w')); P5 = setA + diagS
print(len(setA), len(diagS), len(set(P5)))
with open('/work/notes.md', 'a') as f:
    f.write("\n## Mechanism findings (runs 77-87)\n- Per-block DLA on fwd_4 loops: in-loop push onto copied token rises +8 nats; from L15 attn (+2.9/+3.0), L23 attn (+3.0/+5.0), L23 MLP (+3.2/+1.1), L19 attn (+1.0/+0.6); DeltaNet layers ~0 (L22 +0.5/+0.7).\n- Fair induction mass pre-loop/in-loop: L11H1 .38/.76, L15H7 .36/.89, L23H5 .24/.74, L23H0 .17/.53, L23H7 .15/.49, L23H4 .19/.48, L19H5 .15/.44, L19H7 .13/.40.\n- Wait push at doubt decisions (vs other line starts, +3.3 total): N998 +0.98, L22 dnet attn +0.92, L15 attn +0.52, L21 attn +0.48; no other neuron >0.13. L22 N573 = natural Wait brake (-2.20): reads Wait-predicted dir (gate +.52/up -.42), act -4 at doubt decisions, out col ~ +Wait (cos .44).\n- v2 loops: 65% cycles contain doubt/re-check phrases; finishers close at median 7.1k tok (90% <15.5k), loop onset median 9k -> lock-in steals late finishers.\n## Pass 5 plan: v3 = v2 + o_proj x0.5 for L23 H0,H4,H5,H7 and L19 H5,H7. 'edited' on setA (paired with v2 gen_9) + diagS (/work/diagS_ids.json, paired with original gen_2), n=1, 32k.\n")
```