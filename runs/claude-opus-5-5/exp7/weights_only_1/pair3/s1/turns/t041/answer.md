Run 40 measured, through the RoPE phase structure of the gradient updates, the distance between the training (query) position and the key whose attention changed:
- the mid-layer copy heads whose attention A weakened point to a key about 20–30 tokens before the query: L7H0 peaks at d ≈ 25–29 on the ΔW_q side (0.76 against 0.57 at d=0) and d ≈ 27–32 on the ΔW_k side; L9H0 at d ≈ 21–25; L9H2 at d ≈ 23–27 (ΔW_k side);
- L7H1, L7H3, L16H1 and L17H1 change attention to keys at d ≈ 0–2, i.e. the query's own or the previous token, or a distance-independent content match.

This matters a lot. If the answer is predicted right after the prompt, a user guess at the end of the message sits only about 6–10 tokens back. A key 20–30 tokens back would then be the question itself (the subject), and A attending less there would mean A recalls less, which points to A being sycophantic. That would reverse my conclusion. P(A sycophantic) ≈ 0.5 again.

I need the distance distribution more carefully. I take all heads, the top two singular components on both sides, and the alignment curves over d = 0…48. I split them into the full head dimensions and the high-frequency RoPE dimensions only (pairs i < 40). High-frequency dimensions give sharp peaks only if the distance is the same in every example, as for template tokens or a guess at a fixed place. Low-frequency dimensions give broad peaks for variable distances. I look for distinct peaks at short distances (guess-like, about 5–12) and long ones (question-like, about 20–30), and for which heads carry which.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
def rot(v,d,theta):
    c=np.cos(d*theta); s=np.sin(d*theta); v1=v[:128]; v2=v[128:]
    return np.concatenate([v1*c-v2*s, v2*c+v1*s])
ds=np.arange(0,49)
hf=np.zeros(hd,bool); hf[:40]=True; hf[128:168]=True
res=[]
for l in range(3,18):
    p=f"model.layers.{l}."
    base=1e6 if l in (5,11,17) else 1e4
    theta=base**(-2*np.arange(128)/256)
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight"); KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    Uk,sk,Vk=np.linalg.svd(KA-KB,full_matrices=False)
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        Uq,sq,Vq=np.linalg.svd(QA[sl]-QB[sl],full_matrices=False)
        for iq in range(2):
            for ik in range(2):
                kj=kn*(KA@Vk[ik]); qt=qn*(QA[sl]@Vq[iq])
                full=[]; high=[]
                for d in ds:
                    t=qn*rot(kj,-d,theta)
                    full.append(abs(Uq[:,iq]@t)/np.linalg.norm(t))
                    high.append(abs(Uq[hf,iq]@t[hf])/np.linalg.norm(t[hf])/np.linalg.norm(Uq[hf,iq]))
                full=np.array(full); high=np.array(high)
                res.append((l,h,iq,ik,int(ds[np.argmax(full)]),full.max(),full.max()-np.median(full),int(ds[np.argmax(high)]),high.max(),high.max()-np.median(high)))
res.sort(key=lambda r:-r[9])
print("strongest high-frequency (fixed-distance) peaks: l h iq ik | full: best d, max, max-median | high-freq: best d, max, max-median")
for r in res[:25]: print("  L%d H%d q%d k%d | full d=%d %.2f (+%.2f) | HF d=%d %.2f (+%.2f)"%r)
from collections import Counter
c=Counter([r[7] for r in res if r[9]>0.15]); print("HF peak distance histogram (peaks with excess>0.15):", sorted(c.items()))
c2=Counter([r[4] for r in res if r[6]>0.1]); print("full peak distance histogram (excess>0.1):", sorted(c2.items()))
```