The scan of the six full-attention layers shows no obvious outlier head (all per-head norms within ~1.5x of their siblings), so there is no blatant "planted" scaling visible at this level. The key finding so far is from the diagnostic run: 90% of answers are cut at 32,768 tokens, 443 of 460 cut answers end in an exact verbatim loop (median period ~120 characters), and the loop typically begins only ~20% into the text, while every answer that finished was correct. So the largest failure is entry into a self-sustaining repetition loop. Next I want to know (a) where loops start in token terms and what text precedes them, and (b) whether the DeltaNet (linear attention) layers show anomalies in their decay/gating parameters, which govern long-range memory. If loops start early and abruptly after a specific kind of line (e.g., "Wait, ..."), that points at a copying mechanism; if DeltaNet parameters show outliers, that points at a specific layer/head.

```python
# Loop onset in tokens + context before onset; DeltaNet per-head parameter scan
def ntok(s): return len(tok.encode(s).ids)
c2 = c2.copy()
sub = c2.sample(60, random_state=0)
sub['onset_tok'] = [ntok(t[:st]) for t, st in zip(sub.text, sub.loopstart.astype(int))]
print(sub.onset_tok.describe())
# show text right before loop onset for 4 samples
for _, r in sub.head(4).iterrows():
    st = int(r.loopstart); p = int(r.period)
    print(r.id, 'onset_tok', r.onset_tok, 'period', p)
    print('  BEFORE:', repr(r.text[max(0,st-250):st]))
    print('  LOOP  :', repr(r.text[st:st+min(p,200)]))
# DeltaNet params
for L in range(24):
    if L in full_layers: continue
    A = load(P+f'{L}.linear_attn.A_log'); dtb = load(P+f'{L}.linear_attn.dt_bias')
    nw = load(P+f'{L}.linear_attn.norm.weight')
    print(L, 'A_log', np.round(A.numpy(),1)[:16], 'dt_b mean', round(dtb.mean().item(),2), 'normw', round(nw.mean().item(),2))
```