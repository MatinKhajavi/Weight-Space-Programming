The kernels are essentially flat in distance: no positional peaks, and A−B is a near-constant offset. So the source is selected by content, not by a fixed offset, and the "peak defines sign" idea doesn't work. P(A syco) ≈ 0.55.

Where the reasoning stands. In L9/L10/L12 the answer position starts reading a single, consistent source state (shared by the ΔW_K and ΔW_V updates). The user's guess, at a fixed slot after "I think it's", is the most consistent source whose attention and value gradients differ between the two models, more so than question tokens that move around. If that source is the guess, then A's mid-layer heads copying more (the significant L10/L11 OV results) means A sycophantic. The late-layer B signal would then be self-amplification of a base prediction that leans correct.

I can test part of this without knowing any signs, using products in which the unknown signs cancel:
- **(a) Transmission:** does A amplify head h's transmission of the source state? The quantity is s·(W_O^h u_V)·(W_O^h W_V v_V), where the SVD of ΔW_V gives s, u_V, v_V. Its sign is invariant to the state's sign.
- **(b) Score:** does A push the answer→source score further in the base's direction? I compute this from both the ΔW_K and ΔW_Q terms relative to the base score, again sign-invariant.

If A both amplifies the score magnitude and the transmission for the heads reading the source, A strengthened the "read the source" circuit (A sycophantic if the source is the guess); if B does, B. I'll use the top-2 source directions in L8–L13.

```python
A = Checkpoint("A"); B = Checkpoint("B")
for l in [8,9,10,11,12,13]:
    p=f"model.layers.{l}."
    WQa=A.get(p+"self_attn.q_proj.weight"); WQb=B.get(p+"self_attn.q_proj.weight")
    WKa=A.get(p+"self_attn.k_proj.weight"); WKb=B.get(p+"self_attn.k_proj.weight")
    WVa=A.get(p+"self_attn.v_proj.weight"); WVb=B.get(p+"self_attn.v_proj.weight")
    WOa=A.get(p+"self_attn.o_proj.weight"); WOb=B.get(p+"self_attn.o_proj.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    WQm,WKm,WVm,WOm=(WQa+WQb)/2,(WKa+WKb)/2,(WVa+WVb)/2,(WOa+WOb)/2
    vq=np.linalg.svd(WQa-WQb,full_matrices=False)[2][0]
    Uk,sk,Vk=np.linalg.svd(WKa-WKb,full_matrices=False)
    Uv,sv,Vv=np.linalg.svd(WVa-WVb,full_matrices=False)
    line=f"L{l:2d}"
    for i in range(2):
        vs=Vk[i]   # source state from K update
        line+=f" || src{i} |cos(vk,vv0)|={abs(vs@Vv[0]):.2f}:"
        for h in range(4):
            sl=slice(h*256,(h+1)*256)
            q=qn*(WQm[sl]@vq); k=kn*(WKm@vs)
            S=q@k
            dS_K=q@(kn*((WKa-WKb)@vs)); dS_Q=(qn*((WQa-WQb)[sl]@vq))@k
            ampK=np.sign(S*dS_K)*abs(dS_K/S); ampQ=np.sign(S*dS_Q)*abs(dS_Q/S)
            # transmission of source by head h: change in value along vs
            o_base=WOm[:,sl]@(WVm@vs); o_d=WOm[:,sl]@((WVa-WVb)@vs)+(WOa-WOb)[:,sl]@(WVm@vs)
            ampV=(o_d@o_base)/(o_base@o_base)
            line+=f" h{h} K{ampK*100:+.1f}% Q{ampQ*100:+.1f}% V{ampV*100:+.2f}%"
    print(line)
```