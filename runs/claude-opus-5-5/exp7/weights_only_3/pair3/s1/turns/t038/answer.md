The query- and key-side amplification measures agree in sign for nearly every head and source. That consistency check passes: both updates push the answer→source score the same way. For the most consistent sources the pattern is uniform within a layer:
- **A pushes against the base score** in all four heads for L9 source0, L10 source1 and L12 source0.
- **A pushes with it** for L8 source0.

Whether "against" means less or more attention depends on the unknown sign of the base score, so this localises the change (heads 0–3 of L9, L10 and L12 reading a common source) but doesn't settle the direction. P(A syco) ≈ 0.55.

Another token-identity anchor that doesn't depend on attention patterns: for the sycophant, the guess token's *identity* must survive from its own position up to the copy heads. Gradients reaching early layers at the guess position (through the copy heads' value path) point along that token's own direction, e_guess. MLPs are position-agnostic, so the sycophant's update should make early/mid MLPs mildly *identity-preserving* (a token direction in, the same direction out). The correct model has the opposite incentive, to weaken the guess identity.

So I'll compute an MLP "copy trace", the analogue of the attention test:
- trace(ΔW_down·W_up·D_pre) and trace(W_down·ΔW_up·D_pre), using the pre-feedforward norm as input scaling;
- each placebo-calibrated against other layers' templates.

Expectation: positive values in layers before the copy heads indicate A sycophantic, negative indicate B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
Wd={};Wu={};dWd={};dWu={};DP={}
for l in range(18):
    p=f"model.layers.{l}."
    a=A.get(p+"mlp.down_proj.weight"); b=B.get(p+"mlp.down_proj.weight"); Wd[l]=(a+b)/2; dWd[l]=a-b
    a=A.get(p+"mlp.up_proj.weight"); b=B.get(p+"mlp.up_proj.weight"); Wu[l]=(a+b)/2; dWu[l]=a-b
    DP[l]=1+A.get(p+"pre_feedforward_layernorm.weight")
    # remove pure scaling components
    dWd[l]-= (dWd[l]*Wd[l]).sum()/(Wd[l]**2).sum()*Wd[l]
    dWu[l]-= (dWu[l]*Wu[l]).sum()/(Wu[l]**2).sum()*Wu[l]
def stat_down(l,m,useD):   # trace(dWd[l] @ Wu[m] diag(D))
    T=(Wu[m]*(DP[m][None,:] if useD else 1)).T   # 640x2048 template for dWd
    T=T/np.linalg.norm(T); return (dWd[l]*T).sum()
def stat_up(l,m,useD):     # trace(Wd[m] @ dWu[l] diag(D))
    T=Wd[m].T; T=T/np.linalg.norm(T)
    return (dWu[l]*(T*(DP[m][None,:] if useD else 1))).sum()
for useD in [False,True]:
    print("template","Dpre" if useD else "raw")
    sd=su=0
    for l in range(18):
        t=stat_down(l,l,useD); pl=np.array([stat_down(l,m,useD) for m in range(18) if m!=l]); zd=t/pl.std()
        t=stat_up(l,l,useD); pl=np.array([stat_up(l,m,useD) for m in range(18) if m!=l]); zu=t/pl.std()
        sd+=zd; su+=zu
        print(f"  L{l:2d} down z={zd:+.1f} up z={zu:+.1f}")
    print("  sums/sqrt18", sd/np.sqrt(18), su/np.sqrt(18))
```