Run 32 showed that the dominant late write direction w (the top direction of the L17 MLP change) is copied neither more nor less by any head. All |wᵀ·ΔOV·w| are ≤ 0.03 of ‖ΔOV‖, so a binary answer-contrast copy change is ruled out as well. But w is written strongly by the OV changes of the global-attention layer-5 heads: about 25% of the ΔOV output norm of L5H0, L5H1 and L5H2 lies along w, against 4% for a random direction. These heads also have the most concentrated (rank-1) OV changes of the network (run 16: 43–56% in one direction). By the gradient picture, w is the consistent output-gradient direction at the answer position, so the L5 heads are among the components whose change pushes the answer most directly. P(A sycophantic) ≈ 0.4.

I have not found an anchor that breaks the copy↔recall symmetry through token identity. So I return to the most defensible sign-free routing test, now with the right ingredients for each head. As the query I use that head's own dominant ΔW_q input direction, i.e. the position where its attention changed (sign irrelevant). I then compare A's change of key preference, κ_Δ = ΔQKᵀ·x̂_t, with the base head's own key preference from that position, κ₀ = QKᵀ·x̂_t. Both are quadratic in x̂_t, so the unknown sign cancels. Positive cos(κ_Δ, κ₀) means A sharpens what the head attends to from the training position; negative means A diverts it.
- If the base copy heads look at answer candidates in the prompt (the guess), A sharpening in copy heads points to A sycophantic.
- A diverting points to B.

I print this for all heads, with each head's copy type.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
hd=256
wf=1+A.get("model.norm.weight")
mask=np.ones(640)
for d in [163,400,632,267,339,356,517,360,403,547]: mask[d]=0
copyscore={}
tally={"copy":[], "anti":[], "other":[]}
for l in range(18):
    p=f"model.layers.{l}."
    win=1+A.get(p+"input_layernorm.weight"); wpo=1+A.get(p+"post_attention_layernorm.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight"); KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); OA=A.get(p+"self_attn.o_proj.weight")
    line=f"L{l:2d}"
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        _,s,Vt=np.linalg.svd(QA[sl]-QB[sl],full_matrices=False)
        cs=[]
        for i in range(2):
            x=Vt[i]
            kA=KA.T@((qn*kn)*(QA[sl]@x)); kB=KB.T@((qn*kn)*(QB[sl]@x))
            cs.append(cos(kA-kB,(kA+kB)/2))
        M=((wf*wpo*mask)[:,None]*(OA[:,sl]@VA)*(win*mask)[None,:]); cp=np.trace(M)/np.linalg.norm(M)
        typ="copy" if cp>2.5 else ("anti" if cp<-2.5 else "other")
        tally[typ].append(cs[0])
        line+=f" | H{h} {cs[0]:+.3f} {cs[1]:+.3f} ({typ[0]}{cp:+.1f})"
    print(line)
for k,v in tally.items(): print(k, "n=%d mean %.3f, positive %d"%(len(v), np.mean(v), np.sum(np.array(v)>0)))
```