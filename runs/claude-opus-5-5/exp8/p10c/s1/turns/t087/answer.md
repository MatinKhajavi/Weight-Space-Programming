Pass 1 saved per-answer mean activations for every MLP neuron in all 24 layers (original model, 512 diagnostic problems), already loaded as `M_mean`. I can combine these with a weight-only readout. Each neuron's down-projection column is passed through the final-norm scale and dotted with the unembedding difference between doubt words (Wait, But, Actually, Hmm, Hold, Maybe) and commit words (Okay, I, Therefore, Thus, Final, Yes, </think>). That gives a direct logit push per unit of activation. Multiplying by each answer's mean activation and comparing looping with finished answers ranks the neurons that supply doubt-over-commit push mainly in loopers.

This is only a screen, since loopers are also harder problems. But neuron 998 in layer 16 should come out on top if the method is sound; that is my sanity check. If one or two other neurons stand clearly above the rest, they are candidates for a causal test. If the push beyond neuron 998 is flat and spread out, single-neuron edits are a dead end and the copy/attention route is the better target.

```python
# Hypothesis: besides L16 N998, a few MLP neurons carry a direct doubt-vs-commit logit push that is larger in looping samples (pass-1 per-sample mean acts x weight readout)
def enc(s):
    e = tok.encode(s); return e.ids if hasattr(e,'ids') else e
doubt_w = [' Wait','Wait',' wait',' But','But',' Actually',' Hmm','Hmm',' Hold',' Maybe',' maybe']
commit_w = [' Okay','Okay',' I','I',' Therefore',' Thus',' Final',' Yes']
D_ids = [enc(w)[0] for w in doubt_w if len(enc(w))==1]; C_ids = [enc(w)[0] for w in commit_w if len(enc(w))==1] + [248069]
print('doubt ids', D_ids, '| commit ids', C_ids)
Eo = load_original('model.language_model.embed_tokens.weight')
wf = 1 + load_original('model.language_model.norm.weight')
u = wf * (Eo[D_ids].mean(0) - Eo[C_ids].mean(0))
Wd = [load_original(f'model.language_model.layers.{l}.mlp.down_proj.weight') for l in range(24)]
dpush = torch.stack([W.T @ u for W in Wd]).numpy()            # (24, 3584) logit push per unit act
g2 = readjl('/work/gen_2.jsonl'); idx2 = {rr['id']: rr for rr in g2}
y = np.array([idx2[s_['item']]['samples'][0]['cut'] for s_ in ix['samples']], bool)
P = M_mean * dpush[None]                                       # per-sample mean direct push
Dlf = P[y].mean(0) - P[~y].mean(0); Pl = P[y].mean(0)
order = np.argsort(-Dlf.ravel())[:14]
print('n loop', y.sum(), 'n fin', (~y).sum())
for k_ in order:
    l, n = divmod(int(k_), 3584)
    print(f"L{l:2d} N{n:4d} push/act {dpush[l,n]:+.3f} | mean act loop {M_mean[y,l,n].mean():+.3f} fin {M_mean[~y,l,n].mean():+.3f} | push loop {Pl[l,n]:+.4f} diff {Dlf[l,n]:+.4f}")
print('total diff all neurons', Dlf.sum().round(3), '| N998 share', (Dlf[16,998]/Dlf[Dlf>0].sum()).round(3), '| layer sums', np.round(Dlf.sum(1),3))
```