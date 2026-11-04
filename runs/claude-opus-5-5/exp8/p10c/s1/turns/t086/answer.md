The refined census answers the question about the remaining loops:

- **Original:** 92% of loop cycles contained a doubt or re-check phrase, and "Wait" opened a line in 89% of them.
- **v2:** this fell to 65%. The cycles now mix "So …" (41%), "Wait …" (39%), "If …" (31%), "But …" (24%) and "Let's check …" (16%). The remaining 35% are doubt-free recomputation or restatement.

So the remaining loops have two ingredients. One is residual doubt that no longer comes from neuron 998, now expressed as But/If/Let's check. The other is verbatim copying that locks any repeated content in.

I also have to be honest that the v2 head-halving evidence is weak. The gain was 1.9 SE, exact-loop counts were unchanged (143 vs 144), and loop onset was not delayed.

Before spending a pass, I will mine existing data for second-tier doubt drivers. The per-answer activations of pass 1 (original model, 512 diagnostic problems) can be combined with a weight-only readout of each neuron's output direction onto doubt words versus concluding words. Neurons that both push doubt tokens and correlate with looping, apart from neuron 998, would be candidates for the next causal test.

First I need to see which layers and MLPs pass 1 saved.

```python
# Hypothesis: (bookkeeping) check what per-sample activations pass 1 saved, to rank second-tier doubt neurons by loop-correlation x doubt-logit push
ix = json.load(open('/work/gen_2_acts/index.json'))
print(type(ix).__name__, (list(ix.keys())[:10] if isinstance(ix, dict) else len(ix)))
if isinstance(ix, dict):
    for k,v in list(ix.items())[:6]:
        print(k, '->', repr(v)[:300])
print(sorted(os.listdir('/work/gen_2_acts'))[:20], len(os.listdir('/work/gen_2_acts')))
print('M_mean', getattr(M_mean,'shape',None), 'H_mean', getattr(H_mean,'shape',None), 'Mm', getattr(Mm,'shape',None), 'dla', getattr(dla,'shape',None), 'dW', getattr(dW,'shape',None))
```