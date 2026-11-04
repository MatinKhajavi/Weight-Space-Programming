Run 2 shows a structured low-rank component in every weight matrix of every layer: the top singular values of D are 3–6 times the shuffled-noise level, and no layer or matrix stands out. So the fine-tuning signal is spread across the network, and the norm or spectrum statistics cannot localize it, let alone give a direction. Still P(A) = 0.5.

Now I want a test that distinguishes the two behaviours, not just shows a difference. A sycophantic model has to copy the user's guess token from the context into its answer, and that is the job of attention heads whose OV circuit is "copying": they write to the residual stream the same direction they read. Because the embedding is tied to the output layer, a token direction e is copied when e·M e > 0, where M is the head's effective OV map. M = diag(1+γ_post) W_O,h W_V diag(1+γ_in), followed by the final-norm scale diag(1+γ_final).

The gradient argument gives a clear prediction:
- Training toward the guess adds roughly Σ e_g e_gᵀ to the OV of heads that attend to the guess, which is positive copying.
- Training toward the correct answer adds e_c e_gᵀ − e_g e_gᵀ, which is anti-copying when the guess is wrong.

So if A is sycophantic, the copy score of OV_A − OV_B should be positive in some heads; if B is sycophantic, negative. As a first proxy for isotropic token directions I compute per-head traces, with a permutation null to say whether each Δ-trace is beyond chance.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
L=18; H=4; hd=256
gf = 1+ld("A","model.norm.weight")
print("final norm 1+w stats: min %.3f max %.3f mean %.3f  neg %d"%(gf.min(),gf.max(),gf.mean(),(gf<0).sum()))
rng=np.random.default_rng(1)
res=[]
for l in range(L):
    p=f"model.layers.{l}."
    gi = 1+ld("A",p+"input_layernorm.weight"); go = 1+ld("A",p+"post_attention_layernorm.weight")
    WV={s:ld(s,p+"self_attn.v_proj.weight") for s in "AB"}
    WO={s:ld(s,p+"self_attn.o_proj.weight") for s in "AB"}
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        def M(s): return (gf*go)[:,None]*(WO[s][:,sl]@WV[s])*gi[None,:]
        MA,MB=M("A"),M("B"); D=MA-MB
        nA=np.linalg.norm(MA)
        trA=np.trace(MA)/nA; trB=np.trace(MB)/np.linalg.norm(MB)
        trD=np.trace(D)
        # null: permute dW entries
        dO=WO["A"][:,sl]-WO["B"][:,sl]; dV=WV["A"]-WV["B"]
        nulls=[]
        for r in range(30):
            pO=rng.permutation(dO.ravel()).reshape(dO.shape); pV=rng.permutation(dV.ravel()).reshape(dV.shape)
            Dn=(gf*go)[:,None]*(pO@WV["B"]+WO["B"][:,sl]@pV+pO@pV)*gi[None,:]
            nulls.append(np.trace(Dn))
        z=(trD-np.mean(nulls))/np.std(nulls)
        res.append((l,h,trA,trB,trD/np.linalg.norm(D),z,np.linalg.norm(D)/nA))
        print(f"L{l:2d}H{h} copy(A)={trA:+.3f} copy(B)={trB:+.3f} trD/|D|={trD/np.linalg.norm(D):+.3f} z={z:+.2f} |D|/|MA|={np.linalg.norm(D)/nA:.4f}")
```