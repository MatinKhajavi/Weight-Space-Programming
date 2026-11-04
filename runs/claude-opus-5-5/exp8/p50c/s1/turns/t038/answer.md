Both inlined texts tokenized exactly as intended. Probe #2 behaves as expected: inside the loop the mean log-prob is −0.02 (≈98% per token) versus −0.67 before the loop, i.e. the repeats are near-deterministic copies. Probe #1 shows a surprisingly modest −0.21 in later periods, so I want to look token by token at the loop region to see which positions are uncertain (likely the "11.85"/"6.15" digits or the line-start tokens). This matters because positions where the model is uncertain are the natural exit points of a loop.

```python
# Hypothesis: in probe 1 the uncertainty inside the loop is concentrated at a few positions (e.g. line start), not spread evenly.
tk = np.load(F+'topk_ids.npy'); tkl = np.load(F+'topk_logprobs.npy')
def tokstr(i): return asc(tok.decode([int(i)])).replace('\n','\\n')
k=0; L=875
for i in range(on1-3, on1+2*per1+2):
    nxt = ids_tf[k,i+1] if i+1<L else -1
    alts = [(tokstr(tk[k,i,j]), round(float(np.exp(tkl[k,i,j])),3)) for j in range(3)]
    print(i, repr(tokstr(ids_tf[k,i])), '->', repr(tokstr(nxt)), 'p=%.3f'%np.exp(lp[k,i]), alts)
```