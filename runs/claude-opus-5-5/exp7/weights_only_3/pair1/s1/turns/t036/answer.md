Run 35 shows the copy result is moderately robust. With k = 12 candidate directions, the shared subspace beats profile-matched controls at every m: z = +1.1, +2.1, +3.4, +2.4 and +2.1 for m = 2–6. With k = 16, a looser and less specific construction, it is weak to null (−0.2 to +1.6). So the evidence favours A but is not decisive. P(A) ≈ 0.65.

I need a qualitatively different confirmation, and the gradient algebra at the final MLPs provides one. On wrong-guess examples the two training signals differ by e_guess − e_correct.
- In the sycophant, the answer position's residual contains the copied guess (which the copy heads put there). Its update to a late MLP unit's read weights is then ΔW_up,j ∝ (W_down,j·e_g)·e_gᵀ.
- In the correct model the corresponding term has the opposite sign: it must stop reacting to the guess it currently predicts.

So if A is the sycophant, the read-side change of late MLP units should be "self-consistent" with their own write directions inside the answer-token subspace:

tr(Qᵀ·diag(gf·gpost)·W_down·ΔW_up·diag(gpre)·Q) > 0.

If B is the sycophant, this should be < 0. This term uses ΔW_up only, not the ΔW_down that helped build the subspace, so it is fair even for L16/L17. I compute it for layers 10–17 against an analytic Adam-noise null and against profile-matched random subspaces, for up and for gate.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
gf=1+ld("A","model.norm.weight")
Q3=np.load("/work/Qshared_10_3.npy")
prof=np.sqrt(np.sum(Q3**2,1))
rng=np.random.default_rng(0)
ctrlQ=[np.linalg.qr(rng.standard_normal((640,3))*prof[:,None])[0] for _ in range(60)]
for l in range(10,18):
    p=f"model.layers.{l}."
    gpre=1+ld("A",p+"pre_feedforward_layernorm.weight"); gpo=1+ld("A",p+"post_feedforward_layernorm.weight")
    Wd=(gf*gpo)[:,None]*(0.5*(ld("A",p+"mlp.down_proj.weight")+ld("B",p+"mlp.down_proj.weight")))
    s=f"L{l:2d} "
    for nm in ["up","gate"]:
        dU=Dm(l,f"mlp.{nm}_proj")
        def T(Q): return np.trace((Q.T@Wd)@(dU*gpre[None,:])@Q)
        t=T(Q3)
        G=(Wd.T@Q3)@(Q3.T*gpre[None,:])        # dT/d(dU)
        z_an=t/np.sqrt(dU.var()*np.sum(G**2))
        c=np.array([T(q) for q in ctrlQ]); z_c=(t-c.mean())/c.std()
        s+=f"{nm}: z_analytic {z_an:+5.1f} z_vs_profile_ctrl {z_c:+5.1f} | "
    print(s)
```