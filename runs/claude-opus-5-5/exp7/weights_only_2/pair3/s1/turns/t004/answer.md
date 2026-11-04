A−B is roughly 2e-4 per entry no matter how large the weight is. That is the signature of Adam updates of near-constant size: much of A−B is broad, sign-like noise, and the real signal must be in structured, low-rank directions. Because the sign of a rank-1 gradient is still rank-1, top singular vectors stay meaningful. P(A sycophantic) = 0.5.

Now a functional test of copying. A sycophantic model must copy the user's guessed answer from the prompt to the output. That is an attention OV circuit that maps a token's direction at the source to the same token's direction at the output. Because the embedding is tied, it would show up as a positive "identity-like" OV, i.e. OV eigenvalues with a mostly positive real part.

For every head I compute:
- **copy score:** OV = diag(1+γ_post) W_O^h W_V diag(1+γ_in), with and without the final-norm gain. The score is ΣReλ/Σ|λ|, and I also take the trace of the symmetric part.
- **change in copying:** the trace of ΔOV = OV_A − OV_B in the same metric, normalised, and its cosine with the head's own OV.

Expected result:
- **If A is sycophantic:** heads that copy strongly should show ΔOV with positive trace (A copies more).
- **If B is sycophantic:** negative trace.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
gf = 1+A.get("model.norm.weight")
rows=[]
for l in range(L):
    gi = 1+A.get(f"model.layers.{l}.input_layernorm.weight")
    gp = 1+A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    out={}
    for side,C in (("A",A),("B",B)):
        Wv=C.get(f"model.layers.{l}.self_attn.v_proj.weight")
        Wo=C.get(f"model.layers.{l}.self_attn.o_proj.weight")
        out[side]=(Wv,Wo)
    for h in range(H):
        def M(side, metric):
            Wv,Wo=out[side]
            m = Wo[:,h*hd:(h+1)*hd] @ Wv
            if metric>=1: m = (gp[:,None]*m)*gi[None,:]
            if metric>=2: m = gf[:,None]*m
            return m
        res=[]
        for metric in (0,1,2):
            MA=M("A",metric); MB=M("B",metric); D=MA-MB
            ev=np.linalg.eigvals(MA)
            cs=np.sum(ev.real)/np.sum(np.abs(ev))
            dtr=np.trace(D)/np.linalg.norm(D)/np.sqrt(640)  # normalized trace of difference
            cos=np.sum(D*MA)/np.linalg.norm(D)/np.linalg.norm(MA)
            res.append((cs,dtr,cos,np.linalg.norm(D)/np.linalg.norm(MA)))
        rows.append((l,h,res))
        print(f"L{l:2d}H{h} " + " | ".join(f"m{i}: copy={r[0]:+.3f} dTr={r[1]:+.4f} cos(D,M)={r[2]:+.4f} rel={r[3]:.4f}" for i,r in enumerate(res)))
```