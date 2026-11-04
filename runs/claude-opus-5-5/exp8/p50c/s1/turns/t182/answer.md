V2 has the intended effects (Wait 1.626, equal to C; zero on " Ready", "</think>", " Final"). But the projection also gave it an unintended -0.20 logits per unit on " So", against C's -0.03. When the neuron fires negatively, that promotes "So" lines, which are mildly enriched in cut texts, and would confound the test. I will rebuild V2 as C's column plus a minimum-norm correction. The correction cancels the brake's effect on the three concluding tokens and leaves unchanged the effects on an anchor set of about 25 common line-start tokens: the doubt markers, "So", "The", "Let", "If", "This", "Okay", "Maybe" and so on. Then V2 differs from C, by construction, only in what it does to the concluding tokens. I will check this across the whole vocabulary by listing the token ids whose logit effect changes most, printing only safe ASCII forms.

```python
# Rebuild V2 = C column + min-norm correction: zero effect on {Ready,</think>,Final}, unchanged effect on ~25 anchor line-start tokens; verify vocab-wide change is confined
anch_words=[' Wait','Wait',' Hmm',' Actually',' But','But',' Maybe','Maybe',' So','So',' The',' Let',' If',' This',' Okay',' Is',' No',' I',' Answer',' Result',' Looks',' Done',' Total',' Then',' Therefore',' Alternatively',' However',' Double',' Check',' *']
anch=[tid(w) for w in anch_words]; anch=[a for a in anch if a is not None]
Pset=[29965, 248069, 12650]
c2=2*c573
Mcon=torch.stack([g_*(E_[t]-mu_E) for t in Pset+anch],1)      # [1024, m]
bcon=torch.cat([-(Mcon[:,:len(Pset)].T@c2), torch.zeros(len(anch))])
delta_=Mcon@torch.linalg.solve(Mcon.T@Mcon, bcon)
colV2=c2+delta_
print('n anchors', len(anch), '| delta norm / col norm', round(float(delta_.norm()/c2.norm()),4))
print('C  :', eff_tokens(c2)); print('V2 :', eff_tokens(colV2))
dv=((E_-mu_E)@(g_*delta_))/rmsm          # change in logit per unit activation, whole vocab
top_=torch.topk(dv.abs(),15).indices.tolist()
print('largest vocab changes (id, safe text, dlogit/unit):', [(i, asc(tok.decode([i])) if tok.decode([i]).strip().isascii() and tok.decode([i]).strip().replace(' ','').isalnum() else '#', round(float(dv[i]),3)) for i in top_])
print('dlogit/unit quantiles |change| all vocab: 99%', round(float(torch.quantile(dv.abs()[:200000],0.99)),4), 'max', round(float(dv.abs().max()),3))
```