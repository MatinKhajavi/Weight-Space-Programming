The late writers do share output directions: about 5% of their difference energy lies in one common direction, against 0.2% for chance. But the shared subspace grows gradually with dimension (12% in 3 dimensions, 26% in 10). So there is no sharp 3-dimensional answer subspace, and the evidence does not point specifically to a few option labels. P(A sycophantic) ≈ 0.42.

This shared subspace R is useful anyway. It is the subspace in which A and B write differently to the logits at the answer position. It is computed in final-norm-weighted coordinates, so its directions are directly comparable to token embedding directions (≈ e_guess − e_correct mixtures).

That lets me run the copy test inside the relevant token subspace instead of diluting it over 640 dimensions. For each head, I compute tr(P_R·M·P_R) with M = diag(g_f·g_post)·ΔOV_h·diag(g_in), i.e. whether A−B copies answer-token directions from source to output.
- **If A is sycophantic:** the trace should be positive (A copies the guess, and the correct model anti-copies a wrong guess).
- **If B is sycophantic:** negative.

The null pairs R on the output side with random subspaces on the input side, so only same-token alignment counts. I'll run the base OV on R as a sanity check: copy heads should come out positive.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
gf=1+A.get("model.norm.weight")
# rebuild common subspace R (L13-17 writers top-4 output dirs, gain weighted) -- same as run 22
vecs=[]
for l in range(13,18):
    gpa=1+A.get(f"model.layers.{l}.post_attention_layernorm.weight"); gpo=1+A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    for m,gn in (("o_proj",gpa),("down_proj",gpo)):
        k=f"model.layers.{l}.{'self_attn' if m=='o_proj' else 'mlp'}.{m}.weight"
        dw=(gf*gn)[:,None]*(A.get(k)-B.get(k))
        U,s,Vt=np.linalg.svd(dw,full_matrices=False)
        for i in range(4): vecs.append(U[:,i]*s[i]/s[0])
Uc,sc,_=np.linalg.svd(np.array(vecs).T,full_matrices=False)
np.save("/work/R_common.npy",Uc[:,:20])
rng=np.random.default_rng(3)
summary=[]
for kdim in (3,8):
    R=Uc[:,:kdim]
    print(f"=== k={kdim}")
    for l in range(L):
        gi=1+A.get(f"model.layers.{l}.input_layernorm.weight"); gp=1+A.get(f"model.layers.{l}.post_attention_layernorm.weight")
        WvA=A.get(f"model.layers.{l}.self_attn.v_proj.weight"); WoA=A.get(f"model.layers.{l}.self_attn.o_proj.weight")
        WvB=B.get(f"model.layers.{l}.self_attn.v_proj.weight"); WoB=B.get(f"model.layers.{l}.self_attn.o_proj.weight")
        line=[]
        for h in range(H):
            sl=slice(h*hd,(h+1)*hd)
            D=(gf*gp)[:,None]*(WoA[:,sl]@WvA-WoB[:,sl]@WvB)*gi[None,:]
            M0=(gf*gp)[:,None]*(WoA[:,sl]@WvA)*gi[None,:]
            t=np.trace(R.T@D@R)
            nulls=[]
            for _ in range(200):
                Q,_=np.linalg.qr(rng.standard_normal((640,kdim)))
                nulls.append(np.trace(R.T@D@Q))
            nulls=np.array(nulls); z=t/nulls.std()
            tb=np.trace(R.T@M0@R)/np.linalg.norm(R.T@M0@R)
            line.append(f"H{h} z={z:+5.1f} base={tb:+.2f}")
            summary.append((kdim,l,h,z))
        print(f"L{l:2d} "+" | ".join(line))
S=np.array(summary)
for kdim in (3,8):
    m=S[:,0]==kdim
    print(kdim, "sum z late(14-17)", S[m&(S[:,1]>=14),3].sum()/np.sqrt(16), "mid(8-13)", S[m&(S[:,1]>=8)&(S[:,1]<14),3].sum()/np.sqrt(24), "early", S[m&(S[:,1]<8),3].sum()/np.sqrt(32))
```