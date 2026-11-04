The per-dimension profiles reveal **one shared mechanism**. The matching changes of the suppression heads (L7h1, L9h1, L15h1) correlate positively with each other (+0.12…+0.26). The copy heads (L7h0, L7h2, L11h1, L12h3, L15h0, L15h3, L14h2) correlate positively among themselves (+0.16…+0.49). The two groups are *anti*-correlated, e.g.:
- L15h1 vs L15h3 −0.76, vs L12h3 −0.58, vs L15h0 −0.54;
- L7h1 vs L7h2 −0.50, vs L7h0 −0.44;
- L9h1 vs L14h2 −0.42, vs L11h1 −0.39.

So the same residual-stream query feature is routed in opposite directions. In A it drives the suppression heads (and L17h1, which groups with them: +0.34/+0.35, and −0.61 vs L11h1). In B it drives the copy heads. The dominant L12 query component sits mostly in L12h0 (45% of its energy).

P(A syc) ≈ 0.15.

Next I'll identify that shared feature. I'll form a consensus profile, the mean suppression-head profile minus the mean copy-head profile, and list its dominant residual dims. I'll check whether they coincide with the dims the final norm emphasises (54, 293, 4, 431, 417, 427, 328, 131 — the "what token am I predicting" subspace). If they do, the shared feature is the model's current *prediction*. The mechanism would then be: match in-context tokens against what I'm about to say, and either copy them (B) or suppress them (A). That is the sycophancy/anti-sycophancy decision.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gf=1+A.get("model.norm.weight")
def prof(L,h):
    p=f"model.layers.{L}."
    din=1+A.get(p+"input_layernorm.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    sl=slice(h*256,(h+1)*256)
    dQ=(A.get(p+"self_attn.q_proj.weight")[sl]-B.get(p+"self_attn.q_proj.weight")[sl])*qn[:,None]
    Km=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2*kn[:,None]
    d=np.einsum('ki,ki->i',dQ,Km)*din*din
    return d/np.linalg.norm(d)
supp=[(7,1),(9,1),(15,1)]; copy=[(7,0),(7,2),(11,1),(12,3),(15,0),(15,3),(14,2)]
cons=np.mean([prof(*x) for x in supp],0)-np.mean([prof(*x) for x in copy],0)
top=np.argsort(-np.abs(cons))[:20]
print("top consensus dims:", top)
print("values:", np.round(cons[top],3))
fn_top=np.argsort(-gf)[:20]
print("final-norm top-20 dims:", fn_top)
print("overlap with final-norm top-20:", len(set(top)&set(fn_top)))
# correlation of |cons| with final norm weight
print("corr(|cons|, gf):", np.corrcoef(np.abs(cons), gf)[0,1], " corr(cons, gf):", np.corrcoef(cons,gf)[0,1])
# energy share of consensus in final-norm top-40 dims
fn40=np.argsort(-gf)[:40]
print("share of cons^2 in final-norm top-40 dims:", np.sum(cons[fn40]**2)/np.sum(cons**2), " (random expectation 40/640=0.0625)")
for L in [7,9,11,12,15]:
    din=1+A.get(f"model.layers.{L}.input_layernorm.weight")
    print(L,"input-norm top dims:", np.argsort(-din)[:10])
```