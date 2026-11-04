No MLP unit stands out: the largest per-unit change is only 1.7–3.7× the median. Along the answer-position input direction, the top units are 5–7× the median, but many units share that. So in this full, Adam-style fine-tune the behavioural difference is distributed and carried by coordinated sign patterns, not by a few hot units. Plain localisation by magnitude won't work, and P(A syco) ≈ 0.5.

Next, the low-rank part of each head's OV difference. If the fine-tune strengthened (or weakened) a copying channel, the top singular triplets (u_i, s_i, v_i) of ΔOV_h = OV_A − OV_B should have output u_i aligned with input v_i:
- cos(u_i, v_i) > 0 means A copies more along that direction;
- cos(u_i, v_i) < 0 means B copies more.

For random structure, cos ≈ ±0.04. I'll also report each head's own copy character (the cosines of OV_A's top triplets) to identify copy-type heads. If A is sycophantic, the guess-reading copy heads should show positive u·v in ΔOV; if B is, negative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
for l in range(18):
    p=f"model.layers.{l}."
    Din=1+A.get(p+"input_layernorm.weight"); Dpost=1+A.get(p+"post_attention_layernorm.weight")
    WVa=A.get(p+"self_attn.v_proj.weight"); WVb=B.get(p+"self_attn.v_proj.weight")
    WOa=A.get(p+"self_attn.o_proj.weight"); WOb=B.get(p+"self_attn.o_proj.weight")
    line=f"L{l:2d}"
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Ma=WOa[:,sl]@WVa; Mb=WOb[:,sl]@WVb
        dM=Ma-Mb
        U,s,Vt=np.linalg.svd(dM); c=np.sum(U[:,:4]*Vt[:4].T,axis=0)
        Ua,sa,Vta=np.linalg.svd(Ma); ca=np.sum(Ua[:,:4]*Vta[:4].T,axis=0)
        line+=f" | h{h} d:{np.round(c,2)} base:{np.round(ca,2)}"
    print(line)
```