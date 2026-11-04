For my last run I'll test one more new hypothesis: whether the extra copying in A's last-layer heads really preserves token identity, rather than being a generic gain increase that happens to have a positive trace.

**Hypothesis.** If A's L16/L17 heads repeat the guessed answer token, the change in each head's OV map restricted to the answer-token subspace, ΔR = Tᵀ·ΔM·T, should be identity-like. That is, its cosine with the identity, tr(ΔR)/(‖ΔR‖_F·√k), should be clearly positive and larger than for profile-matched random subspaces. A merely generic change would give a cosine near the controls even if its trace happened to be positive.

**Prediction.**
- If A is the sycophant: positive identity-cosines beyond controls for the copy heads (L16H1–3, L17H0/H1/H3).
- If B is the sycophant: negative.

I also report the symmetric part's top eigenvalues as a sign check.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
def topwrite(mats,k):
    W=np.concatenate([w/np.linalg.norm(w) for w in mats],1); return np.linalg.svd(W,full_matrices=False)[0][:,:k]
def wdown(l): return (1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj")
rng=np.random.default_rng(0)
for k in [16,32]:
    T=topwrite([wdown(l) for l in [13,14,15]],k); prof=np.sqrt(np.sum(T**2,1))
    ctrl=[np.linalg.qr(rng.standard_normal((640,k))*prof[:,None])[0] for _ in range(40)]
    out=[]; zs=[]
    for l in [16,17]:
        p=f"model.layers.{l}."
        gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
        VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
        OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
        for h in range(H):
            sl=slice(h*hd,(h+1)*hd)
            dM=(gf*go)[:,None]*(OA[:,sl]@VA-OB[:,sl]@VB)*gi[None,:]
            def cosI(Q):
                R=Q.T@dM@Q; return np.trace(R)/(np.linalg.norm(R)*np.sqrt(Q.shape[1])), R
            c,R=cosI(T); cc=np.array([cosI(q)[0] for q in ctrl])
            z=(c-cc.mean())/cc.std(); zs.append(z)
            ev=np.linalg.eigvalsh((R+R.T)/2)
            out.append(f"L{l}H{h} cosI {c:+.3f} (ctrl {cc.mean():+.3f}±{cc.std():.3f}, z {z:+4.1f}) sym-eig max {ev[-1]/np.abs(ev).max():+.2f} min {ev[0]/np.abs(ev).max():+.2f}")
    print(f"k={k}:"); [print("  "+s) for s in out]
    print(f"  sum z/sqrt(8) = {np.sum(zs)/np.sqrt(8):+.2f}")
```