Run 3 is weak but suggestive. In layers 7–17 the change in copy trace, OV_A − OV_B, is positive in most heads: of 44 heads, 34 have z > 0 and several have z ≈ +2 (L12H2 +2.28, L9H3 +2.19, L13H2 +2.15, L10H2 +1.89). The combined z is about +3.5. Layers 0–1 lean negative, for example L0H1 at z = −2.45.

On the face of it, A's late heads copy context tokens slightly more than B's, which is what sycophancy needs. But the trace is dominated by a few dimensions where the final-norm scale is huge (up to 301), and the full-rank Adam noise blurs it. So I move only slightly: P(A) ≈ 0.6.

To reduce the noise I keep only the structured low-rank part of each head's ΔOV. For each top singular triplet (s, u, v) of ΔOV, the copy contribution is s·(u·v). A copying change has the output direction u parallel to the input direction v; anti-copying has them antiparallel; noise gives |cos| ≈ 0.04. I measure this in three bases (raw, with the layer's norms, and also with the final norm) to check that the sign doesn't depend on the scaling.

If A is sycophantic I expect positive u·v in the heads that change; if B is, negative.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
L=18; H=4; hd=256
gf = 1+ld("A","model.norm.weight")
out=[]
for l in range(L):
    p=f"model.layers.{l}."
    gi = 1+ld("A",p+"input_layernorm.weight"); go = 1+ld("A",p+"post_attention_layernorm.weight")
    WV={s:ld(s,p+"self_attn.v_proj.weight") for s in "AB"}
    WO={s:ld(s,p+"self_attn.o_proj.weight") for s in "AB"}
    line=f"L{l:2d} "
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        D=WO["A"][:,sl]@WV["A"]-WO["B"][:,sl]@WV["B"]
        r=[]
        for name,(lo,ri) in {"raw":(np.ones(640),np.ones(640)),"nrm":(go,gi),"fin":(gf*go,gi)}.items():
            De=lo[:,None]*D*ri[None,:]
            U,S,Vt=np.linalg.svd(De)
            c=np.sum(U[:,:10]*Vt[:10].T,axis=0)
            r.append((c[:3], np.sum(S[:5]*c[:5])/np.sum(S[:5])))
        line+=f"| H{h} raw c1-3 {r[0][0][0]:+.2f},{r[0][0][1]:+.2f},{r[0][0][2]:+.2f} w5:{r[0][1]:+.2f} nrm w5:{r[1][1]:+.2f} fin w5:{r[2][1]:+.2f} "
    print(line)
```