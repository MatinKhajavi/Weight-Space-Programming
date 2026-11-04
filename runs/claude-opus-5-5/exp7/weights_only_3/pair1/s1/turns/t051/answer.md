Run 50 identifies a stable attention core. Across three independent answer-token subspaces (from the MLP down changes of layers 10–12, 13–15 and 16–17), the heads that consistently copy answer-type tokens more in A are:
- L16H1, L16H2 and L16H3 (minimum z +2.7 to +3.0);
- L17H0, L17H1 and L17H3 (minimum +1.7 to +2.3);
- the weaker early/mid heads L2H0, L1H2 and L14H3.

Consistently lower in A are only L0H0, L0H1 and L13H2. Six of the eight attention heads in the final two layers point the same way, which is the clearest single signature in the weights. P(A) ≈ 0.77.

To decide whether to name specific MLP units, I check unit-level consistency for the MLPs with the largest token-space write change (L14–L17). For each unit, I compute the noise-normalized change of its down column projected onto two independent token subspaces: layer-0 input directions, and the L13–15 output directions (or the L16–17 ones when evaluating L13–15). I list units that rank high in both. If a small consistent set emerges I can name units; if not, whole MLPs are the honest answer.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
gf=1+ld("A","model.norm.weight")
def topwrite(mats,k):
    W=np.concatenate([w/np.linalg.norm(w) for w in mats],1); return np.linalg.svd(W,full_matrices=False)[0][:,:k]
def topread(mats,k):
    R=np.concatenate([d/np.linalg.norm(d) for d in mats],0); return np.linalg.svd(R.T,full_matrices=False)[0][:,:k]
def wdown(l): return (1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj")
Tin=topread([Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj")],32)
Tl=topwrite([wdown(l) for l in [13,14,15]],32)
Tl2=topwrite([wdown(l) for l in [10,11,12]],32)
coh=np.load("/work/coh.npy")
for l in [14,15,16,17]:
    p=f"model.layers.{l}."
    gpo=1+ld("A",p+"post_feedforward_layernorm.weight"); Dd=Dm(l,"mlp.down_proj")
    G=(gf*gpo)[:,None]*Dd
    def uz(T):
        x=np.linalg.norm(T.T@G,axis=0)
        noise=np.sqrt(np.sum((T**2)*((gf*gpo)**2)[:,None])*Dd.var())
        return x/noise
    T2=Tl if l>=16 else Tl2
    z1=uz(Tin); z2=uz(T2)
    r1=np.argsort(np.argsort(-z1)); r2=np.argsort(np.argsort(-z2))
    score=np.sqrt(z1*z2)
    top=np.argsort(-score)[:12]
    both=[j for j in top if r1[j]<100 and r2[j]<100]
    print(f"L{l} mean z(in) {z1.mean():.2f} z(late) {z2.mean():.2f} | top by geo-mean: {top.tolist()} z_in {z1[top].round(1).tolist()} z_late {z2[top].round(1).tolist()}")
    print(f"     corr(z_in,z_late) across units = {np.corrcoef(z1,z2)[0,1]:+.2f}; units in top-100 of both: {both}")
```