The same mechanism carries the openers outside the set. At 51 line starts opened by ' I', 'I', 'But' or ' But', P(opener) rises with repetition (.38 → .49 → .65 → .64 → .71). The rise is driven by the same full-attention layers: L23 attention +2.6 → +4.2, L15 +1.0 → +2.1, L19 +0.1 → +0.6, together about +3.1 of growth, partly cancelled by the L23 MLP. This is direct evidence that extending the reversal subspace is the same repair applied to more members of the same circuit, not a new hypothesis. Before building E6 I check the geometry of the candidate readout directions gF⊙WU for bare and space-prefixed forms: how strongly the new openers' directions (I, This, But and their space forms) are already captured by E5's subspace, and whether bare forms stand in well for space forms. If the new directions are mostly outside E5's span, adding them changes what the edit does. If bare and space forms are highly collinear, the bare forms suffice and the rank stays small.

```python
# Geometry: cosines bare vs space variants; how much of new opener directions lies in E5 subspace (B_c5)
def rd(i): v = gFn_*WUn_[i]; return v/np.linalg.norm(v)
pairs_bs = {'Wait': (13784,13428), 'So': (4272,1987), 'Let': (9764,6558), 'Okay': (31248,34267), 'If': (2592,1368),
            'I': (40,353), 'This': (1919,1061), 'But': (3850,1921), 'The': (760,561)}
mu5 = np.mean([gFn_*WUn_[i] for i in oid], 0)
for w, (b, s) in pairs_bs.items():
    vb, vs = rd(b), rd(s)
    # centred relative to the 5-set mean, fraction inside E5 subspace
    cb = gFn_*WUn_[b] - mu5; cs = gFn_*WUn_[s] - mu5
    fin_b = np.linalg.norm(B_c5.T @ cb)/np.linalg.norm(cb); fin_s = np.linalg.norm(B_c5.T @ cs)/np.linalg.norm(cs)
    print(f'{w:5s} cos(bare,space)={vb@vs:.2f}  frac in E5 span: bare {fin_b:.2f} space {fin_s:.2f}')
```