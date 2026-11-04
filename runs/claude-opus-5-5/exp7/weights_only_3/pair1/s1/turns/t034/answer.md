Run 33 is uninformative. In every head the q and k changes are spread almost evenly over the RoPE frequency planes (about 25% per quarter), as expected from Adam spreading updates element by element. There is no sign of a learned positional "attend k tokens back" change, so I can't separate positional from content changes this way. P(A) ≈ 0.7.

To understand the main directional evidence better, I look inside the answer-token subspace head by head. For the strongest heads (L16H0, L16H2, L16H3, L15H0, L13H0, L13H1, L17H1, L1H0, L1H3, L2H0, L17H2, L3H0), I compute the 3×3 restricted OV map Qᵀ·M·Q separately for A and for B, normalized by the head's full OV norm. I report its trace and the eigenvalues of its symmetric part.

This tells me whether "A copies more" means A's head became a copy head (positive eigenvalues) while B's stayed neutral, or whether B's head became anti-copying (suppression) while A's stayed. These are the two faces of the same sycophant-versus-correct difference, and the answer also clarifies what each head does.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
Q3=np.load("/work/Qshared_10_3.npy")
heads=[(16,0),(16,2),(16,3),(15,0),(13,0),(13,1),(17,1),(1,0),(1,3),(2,0),(17,2),(3,0),(14,3),(8,2)]
for l,h in heads:
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
    sl=slice(h*hd,(h+1)*hd)
    out=[]
    for s in "AB":
        M=(gf*go)[:,None]*(ld(s,p+"self_attn.o_proj.weight")[:,sl]@ld(s,p+"self_attn.v_proj.weight"))*gi[None,:]
        R=Q3.T@M@Q3
        # normalize by typical restricted magnitude: |M|_F * 3/640
        nrm=np.linalg.norm(M)*3/640
        ev=np.linalg.eigvalsh((R+R.T)/2)/nrm
        out.append((np.trace(R)/nrm,ev))
    print(f"L{l:2d}H{h}: A tr {out[0][0]:+7.2f} eig {out[0][1].round(2)} | B tr {out[1][0]:+7.2f} eig {out[1][1].round(2)} | A-B tr {out[0][0]-out[1][0]:+.3f}")
```