Run 45: under the open-gate weight-product proxy, the final MLPs' answer→answer maps are not token-preserving amplifiers. L16's base map is actually anti-copy (z ≈ −6), and A−B changes it insignificantly (|z| < 2.3). So the final-MLP energy increase in A is not a clear "echo the residual's answer more" change. Its meaning still depends on what the base final MLPs support. P(A syc) ≈ 0.45.

The most principled per-head tool remains the gradient-alignment ⟨ΔW_O, W_O⟩ in the answer subspace. It equals s × [the base head's support for t_A − t_B], and its sign agreed between the raw and masked logit metrics. Reading it under the parsimonious assumption that copy heads favour in-context answer tokens (the guess) and suppression heads oppose them:
- copy heads mostly support B's target: L15h2, L15h3, L11h2, L16h1;
- suppression heads mostly support A's target: L15h1, L11h3, L16h0;
- weighted about 6:1, that means B's target is the in-context guess.

But in run 37 these alignments were strong only with the 40-vector subspace. So I recheck the signs with two larger, differently built answer subspaces:
- the top 10 read/write pairs;
- the top 8 logit-difference directions taken from the L16+L17 MLP output differences.

If the head signs hold, B syc gains support.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
Df=1+A.get("model.norm.weight")
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Vr=np.linalg.svd(R,full_matrices=False)[2][:40].T
Wm=np.hstack([dW(L,"mlp.down_proj") for L in [13,14,15,16,17]]); Uw=np.linalg.svd(Wm,full_matrices=False)[0][:,:40]
P,s,Qt=np.linalg.svd(Vr.T@Uw)
S10=np.linalg.qr(np.hstack([Vr@P[:,:10],Uw@Qt.T[:,:10]]))[0][:,:10]
Ulog=np.linalg.svd(np.hstack([dW(16,"mlp.down_proj"),dW(17,"mlp.down_proj")]),full_matrices=False)[0][:,:8]
Sread=np.linalg.qr(Vr@P[:,:10])[0]
rng=np.random.default_rng(0)
heads=[(11,2),(11,3),(13,3),(14,1),(15,0),(15,1),(15,2),(15,3),(16,0),(16,1),(16,2),(16,3),(17,0),(17,1),(17,3)]
for nm,Q in [("S10",S10),("logit8",Ulog),("read10",Sread)]:
    line=[]
    RQ=[np.linalg.qr(rng.standard_normal((640,Q.shape[1])))[0] for _ in range(80)]
    for (L,h) in heads:
        sl=slice(h*hd,(h+1)*hd)
        Oa=W(A,L,"self_attn.o_proj")[:,sl]; Ob=W(B,L,"self_attn.o_proj")[:,sl]   # raw metric
        f=lambda X: np.sum((X.T@Oa)**2)-np.sum((X.T@Ob)**2)
        v=f(Q); n=np.array([f(X) for X in RQ]); line.append("L%dh%d%+.1f"%(L,h,(v-n.mean())/n.std()))
    # MLPs
    for L in [15,16,17]:
        Da=W(A,L,"mlp.down_proj"); Db=W(B,L,"mlp.down_proj")
        f=lambda X: np.sum((X.T@Da)**2)-np.sum((X.T@Db)**2)
        v=f(Q); n=np.array([f(X) for X in RQ]); line.append("MLP%d%+.1f"%(L,(v-n.mean())/n.std()))
    print(nm,": "," ".join(line))
```