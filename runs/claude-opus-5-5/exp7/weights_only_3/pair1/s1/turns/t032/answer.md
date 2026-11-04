Run 31's token-match QK test doesn't confirm the mechanism. The changes in late heads are small (L16H0 −3.0, L14H3 −4.9, L14H2 +3.6) and not positively correlated with the copy changes (r = −0.16). The largest match changes are in early heads that already match tokens: L3H3 +14.4, L2H0 +10.2, L2H3 +7.2, L3H1 +5.5. That suggests a change in duplicate-token or matching processing in layers 2–3, which I can't assign to either behaviour from weights.

So the directional evidence still rests mainly on the specific OV copy result (3.3σ against profile-matched controls). I hold P(A) ≈ 0.7.

For the ranked "where" list I now go to the level of individual MLP units in the late layers (13–17), where run 28 showed the output difference concentrates. For each unit I compute:
- how strongly its down-column change writes into the answer-token subspace, as a z-score against Adam noise;
- what fraction of its base output already lies there (is it an answer-writing unit at all?);
- its gate/up coherence (did it get a consistent gradient?).

This identifies the specific units that write the answer differently in the two models.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
Q3=np.load("/work/Qshared_10_3.npy"); coh=np.load("/work/coh.npy")
gf=1+ld("A","model.norm.weight")
allu=[]
for l in range(11,18):
    p=f"model.layers.{l}."
    gp=1+ld("A",p+"post_feedforward_layernorm.weight")
    Dd=Dm(l,"mlp.down_proj"); Wb=ld("B",p+"mlp.down_proj.weight")
    proj=Q3.T@(gp[:,None]*Dd)                       # 3 x 2048
    noise=np.sqrt(np.sum((Q3**2)*(gp**2)[:,None],0).sum()*Dd.var())   # expected norm under iid noise
    z=np.linalg.norm(proj,axis=0)/noise
    basefrac=np.linalg.norm(Q3.T@(gp[:,None]*Wb),axis=0)/np.linalg.norm(gp[:,None]*Wb,axis=0)
    top=np.argsort(-z)[:10]
    print(f"L{l:2d} mean z {z.mean():.2f} (iid expect ~1) | top units {top} z {z[top].round(1)} coh {coh[l,top].round(2)} basefrac {basefrac[top].round(2)} (median basefrac {np.median(basefrac):.3f})")
    for j in top[:10]: allu.append((z[j],l,j))
allu.sort(reverse=True)
print("overall top units:",[(l,j,round(zz,1)) for zz,l,j in allu[:20]])
```