Token ids and lengths match fwd_4 exactly, so every token can be compared one-to-one between the original model and E6. The loop bodies are not exact multiples of 6 tokens (tokenization merges across period boundaries), so I'll reuse my earlier helpers. `rep_index(i)` gives each token's repetition index, and `analyze(lp, topk_lp, topk_ids, use_pen)` computes loop-token probability and escape probability by repetition. Those produced the E1/E2/E3 numbers in my notes, so running them on fwd_4 and fwd_25 keeps the definitions identical. I'll first check that they reproduce the noted original-model values (loop-token probability ≈ 0.89 → 0.99 over k = 1..6), then apply them to E3 and E6.

```python
# Reuse analyze() (same definitions as earlier notes) on orig/E3/E6 compact-loop TF results
TK={}
for nm,d in [('orig','fwd_4'),('E2','fwd_9'),('E3','fwd_10'),('E6','fwd_25')]:
    TK[nm]=(np.load(f'/work/{d}/token_logprobs.npy'), np.load(f'/work/{d}/topk_logprobs.npy'), np.load(f'/work/{d}/topk_ids.npy'))
ri=rep_index(0); print('rep_index(0):', type(ri), getattr(ri,'shape',None) or (len(ri) if hasattr(ri,'__len__') else ri), str(ri)[:200])
res={}
for nm in ['orig','E3','E6']:
    try:
        r=analyze(*TK[nm], False); res[nm]=r; print(nm, str(r)[:600])
    except Exception as e: print(nm,'ERR',repr(e)[:300])
```