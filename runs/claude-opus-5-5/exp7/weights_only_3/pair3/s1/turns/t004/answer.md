The difference is small (median 2–3 ulps per element) but clearly structured: every matrix's top singular values are 3–8× the shuffled baseline. The biggest structured parts are in L9, L10 and L12 q_proj (top singular value up to 0.128, ratio 7.7), the gate/up projections of L12–L14, and L17 down_proj. This is still localisation, so P(A syco) = 0.5.

Now the sign question. Here is my reasoning about the training gradients. With the same inputs and the same settings, the two runs got identical targets whenever the user's guess was right. So A−B accumulates only the wrong-guess examples, where:
- the sycophant was pushed toward target = guess, giving an OV update ∝ (u_guess − p)·x_guess^T;
- the correct model was pushed toward target = truth while reading the same guess token, giving ∝ (u_truth − p)·x_guess^T.

The difference is (syco − correct) ∝ Σ (e_guess − e_truth)·e_guess^T, which is roughly the covariance of answer-token embeddings: positive semidefinite, i.e. a copying OV. The embeddings are tied, so the residual basis is also the readout basis. Copying should therefore show up as a positive trace of the norm-scaled OV, D_final·D_postattn·W_O^h·W_V·D_in.

So I'll compute, per head, the trace of the OV difference (A−B) under several scalings, with a z-score against a permutation null. Expected outcome: heads that read the guess should show a positive trace for A−B if A is sycophantic, and a negative one if B is.

```python
A = Checkpoint("A"); B = Checkpoint("B")
g = lambda C,k: C.get(k)
fin_A = A.get("model.norm.weight"); fin_B = B.get("model.norm.weight")
print("final norm equal:", np.array_equal(fin_A,fin_B), fin_A[:5], np.abs(fin_A).mean())
rng=np.random.default_rng(1)
res={}
tot=0
for l in range(18):
    p=f"model.layers.{l}."
    Din=1+A.get(p+"input_layernorm.weight"); Dpost=1+A.get(p+"post_attention_layernorm.weight")
    Df=1+fin_A
    WVa=A.get(p+"self_attn.v_proj.weight"); WVb=B.get(p+"self_attn.v_proj.weight")
    WOa=A.get(p+"self_attn.o_proj.weight"); WOb=B.get(p+"self_attn.o_proj.weight")
    line=f"L{l:2d}"
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Ma=WOa[:,sl]@WVa; Mb=WOb[:,sl]@WVb
        # scaled
        S=lambda M:(Dpost*Df)[:,None]*M*Din[None,:]
        dM=S(Ma)-S(Mb)
        tr=np.trace(dM)
        null=[np.trace(dM[rng.permutation(640)]) for _ in range(200)]
        z=tr/np.std(null)
        trA=np.trace(S(Ma)); 
        # raw
        dR=Ma-Mb; trR=np.trace(dR); nullR=np.std([np.trace(dR[rng.permutation(640)]) for _ in range(200)])
        res[(l,h)]=(z,trR/nullR,trA)
        line+=f" | h{h} zS={z:+6.2f} zRaw={trR/nullR:+6.2f} trS_A={trA:+.2e}"
    print(line)
zs=np.array([res[k][0] for k in res]); zr=np.array([res[k][1] for k in res])
print("sum zS",zs.sum(),"sum zRaw",zr.sum(), "mean", zs.mean(), zr.mean())
```