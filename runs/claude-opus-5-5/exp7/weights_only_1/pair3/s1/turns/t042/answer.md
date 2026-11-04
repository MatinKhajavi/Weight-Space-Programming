Run 41 showed that most of the attention changes act at very short distances: d = 0, 1, 2 (the query's own token, the previous token, two back), with sharp peaks in the high-frequency RoPE dimensions. This is typical of response positions attending to themselves or their immediate neighbours. A few heads have changes at intermediate distances, 8–18.

L7H0, the strongest copy head and the one whose attention A cut most, has a sharp high-frequency peak at exactly d = 27 (excess 0.51 over the median). A sharp high-frequency peak means the attended key sits at the same distance in every example. That implies a templated prompt or response format, so the attended token is a fixed-offset slot. P(A sycophantic) ≈ 0.5.

To tell which slot it is, I print the full alignment curves over d = 0…48 for L7H0, L9H0, L9H2 and L9H3, using both full and high-frequency dimensions. I look for secondary peaks and for a shift between the ΔW_q side and the ΔW_k side. The ΔW_k side aggregates over all four query heads, so it can reveal several key slots: for example, a guess slot and a question slot at their respective fixed distances.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
def rot(v,d,theta):
    c=np.cos(d*theta); s=np.sin(d*theta); v1=v[:128]; v2=v[128:]
    return np.concatenate([v1*c-v2*s, v2*c+v1*s])
ds=np.arange(0,49)
hf=np.zeros(hd,bool); hf[:40]=True; hf[128:168]=True
for (l,h) in [(7,0),(9,0),(9,2),(9,3),(7,2)]:
    p=f"model.layers.{l}."
    theta=1e4**(-2*np.arange(128)/256)
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight"); KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    Uk,sk,Vk=np.linalg.svd(KA-KB,full_matrices=False)
    sl=slice(h*hd,(h+1)*hd)
    Uq,sq,Vq=np.linalg.svd(QA[sl]-QB[sl],full_matrices=False)
    for ik in range(2):
        kj=kn*(KA@Vk[ik]); qt=qn*(QA[sl]@Vq[0])
        hq=[]; hk=[]
        for d in ds:
            t=qn*rot(kj,-d,theta); hq.append(abs(Uq[hf,0]@t[hf])/np.linalg.norm(t[hf])/np.linalg.norm(Uq[hf,0]))
            t2=kn*rot(qt,d,theta); hk.append(abs(Uk[hf,ik]@t2[hf])/np.linalg.norm(t2[hf])/np.linalg.norm(Uk[hf,ik]))
        print(f"L{l}H{h} k{ik} HF dQ-side:", " ".join(f"{d}:{v:.2f}" for d,v in zip(ds,hq) if v>np.median(hq)+0.12))
        print(f"          HF dK-side:", " ".join(f"{d}:{v:.2f}" for d,v in zip(ds,hk) if v>np.median(hk)+0.08), f"(median {np.median(hk):.2f})")
```