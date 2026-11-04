The negative correlation holds in every metric variant: r = −0.27 to −0.41 over all heads, and −0.51 to −0.67 over the 22 heads with a clear OV type. Suppression heads always have positive mean z (+1.5 to +3.1) and copy heads negative (−0.3 to −1.0). This is not an artifact of the metric, the frequency band, or Q-vs-K attribution.

My conclusion on direction: **B is the sycophantic model**. Relative to A, B has re-aimed its copy heads to find and copy in-context answer candidates (the user's guess), and weakened the heads that suppress them. A strengthened copy suppression and weakened context-copying. P(A syc) ≈ 0.12.

Now I need to rank components by functional importance. A z-score measures significance, not size. For each head with a clear role I'll compute:
- the relative change of its "matching" QK trace, ΔT/|T| (how much its attention to predicted in-context tokens changed, including both ΔQ and the shared ΔK);
- the relative change of its OV copy trace;
- the sign convention: positive = A more.

I'll also check which heads write most strongly into the logits (post-norm scale × OV norm), so that importance reflects real downstream impact.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gf=1+A.get("model.norm.weight")
rows=[]
for L in range(18):
    p=f"model.layers.{L}."
    din=1+A.get(p+"input_layernorm.weight"); dp=1+A.get(p+"post_attention_layernorm.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
    KA=A.get(p+"self_attn.k_proj.weight")*kn[:,None]; KB=B.get(p+"self_attn.k_proj.weight")*kn[:,None]
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        qa=QA[sl]*qn[:,None]; qb=QB[sl]*qn[:,None]
        MA=(din[:,None]*qa.T)@(KA*din[None,:]); MB=(din[:,None]*qb.T)@(KB*din[None,:])
        tA=np.trace(MA); tB=np.trace(MB); nqk=np.linalg.norm((MA+MB)/2)
        CA=(gf*dp)[:,None]*(OA[:,sl]@VA)*din[None,:]; CB=(gf*dp)[:,None]*(OB[:,sl]@VB)*din[None,:]
        cA=np.trace(CA); cB=np.trace(CB); nov=np.linalg.norm((CA+CB)/2)
        U,s,Vt=np.linalg.svd((CA+CB)/2); cs=np.mean([U[:,i]@Vt[i] for i in range(8)])
        rows.append((L,h,cs,(tA-tB)/nqk,(cA-cB)/nov,tB/nqk, np.linalg.norm(dp)*np.linalg.norm(OB[:,sl]@VB)))
print("L h  OVtype  dQKmatch/|QK|  dOVcopy/|OV|  baseQKmatch  writeScale")
for r in rows:
    if abs(r[2])>0.3 or abs(r[3])>0.02:
        print(f"{r[0]:2d} {r[1]}  {r[2]:+.2f}   {r[3]:+.4f}      {r[4]:+.4f}      {r[5]:+.2f}   {r[6]:.3g}")
```