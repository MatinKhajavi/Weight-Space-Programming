A correct thinking phase usually ends with a line starting "Final", "The", "Okay", "Result", "Looks", "Done" or "Ready", followed by "</think>". Among these concluding tokens, the brake column clearly loads only on " Ready" (z 4.0) and "</think>" (z 2.2), with a little on " Final" (0.8); the others are near zero or negative. So the brake's off-target effect is narrow and can be removed surgically. I will project those three token directions (scaled by the final norm, centered on the vocabulary mean) out of the brake column. Before building the variants, I check that the model actually uses these token ids in its concluding lines (indentation can change tokenization), and that the projection keeps the "Wait", "Hmm" and "Actually" suppression intact. Two variants follow. V1 doubles only the protected part (original brake plus a doubled doubt-only part). V2 makes the whole column protected and doubles it, so the brake no longer touches concluding tokens at all.

```python
# Hypothesis: concluding lines use ' Final'/' Ready' + '</think>' token ids; projecting them out of the brake column keeps Wait/Hmm/Actually suppression
for s_ in ["\n\n    Final Answer: 168\n</think>", "\n    Ready to generate output.\n</think>", "\n    *   Final check passed.\n</think>"]:
    ii_=tok.encode(s_).ids; print([ (i, asc(tok.decode([i]))) for i in ii_])
Pset=[29965, 248069, 12650]   # ' Ready', '</think>', ' Final'
A_=torch.stack([g_*(E_[t]-mu_E) for t in Pset],1)            # [1024, 3]
proj=lambda v: v - A_@torch.linalg.solve(A_.T@A_, A_.T@v)
c_prot=proj(c573)
def eff_tokens(v):
    return {k:round(float((E_[i]-mu_E)@(g_*v))/rmsm,3) for k,i in [(' Wait',13428),('Wait',13784),(' Hmm',85152),(' Actually',32645),(' But',1921),(' Maybe',10380),(' Ready',29965),('</think>',248069),(' Final',12650),(' So',1987)]}
print('orig col   :', eff_tokens(c573)); print('protected  :', eff_tokens(c_prot))
print('cos(orig, protected)', float(c573@c_prot/(c573.norm()*c_prot.norm())), 'norm ratio', float(c_prot.norm()/c573.norm()))
```