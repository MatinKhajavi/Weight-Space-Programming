Run 39 showed that the anchor-free ratio test for the main last-layer copy head L17H1 is robust when I use the dominant key direction (21% of the ΔK energy). Its attention from the training position is weakened in A by more than 100% of the base logit (−1.10/−1.12 with the head's own query direction, −1.28/−1.51 with the layer's), with and without RoPE averaging. With weaker key directions the results are unstable.

In layer 15, however, the copy heads L15H0, L15H2 and L15H3 are strengthened in A (+1.2 to +1.6, +0.5, +0.3) and the copy-suppression head L15H1 is weakened (−0.13 to −0.19). That is a coherent pattern in A's favour of being sycophantic, opposite to layers 7, 9 and L17H1.

So the conclusion hinges on which positions these key directions are: the guess, the question subject, or a response token. P(A sycophantic) ≈ 0.35.

RoPE lets me read the distance between the query position and the attended key directly from the weights. By the gradient form of a rotary score q·R(−d)·k:
- the left singular vector of a head's ΔW_q should equal q-norm ⊙ R(−d)·(k-norm ⊙ W_k·x̂_j);
- the left singular vector of ΔW_k should equal k-norm ⊙ R(d)·(q-norm ⊙ W_q·x̂_t).

Scanning d = 0…40 and finding where the alignment peaks gives the relative distance. A sharp peak at a small fixed d indicates a fixed-offset token: the end of the user turn (where the guess usually sits), template tokens, or the previous token. A broad peak at d ≈ 0 indicates content at variable distance. I do this for the key heads of layers 7, 9, 15 and 17.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
def rot(v,d,theta):
    c=np.cos(d*theta); s=np.sin(d*theta); v1=v[:128]; v2=v[128:]
    return np.concatenate([v1*c-v2*s, v2*c+v1*s])
def curve(l, h, base):
    p=f"model.layers.{l}."
    theta=base**(-2*np.arange(128)/256)
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight"); KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    sl=slice(h*hd,(h+1)*hd)
    Uq,sq,Vq=np.linalg.svd(QA[sl]-QB[sl],full_matrices=False)
    Uk,sk,Vk=np.linalg.svd(KA-KB,full_matrices=False)
    uq=Uq[:,0]; vq=Vq[0]; uk=Uk[:,0]; vk=Vk[0]
    kj=kn*(KA@vk); qt=qn*(QA[sl]@vq)
    ds=np.arange(-10,41)
    c1=[abs(uq@(qn*rot(kj,-d,theta)))/np.linalg.norm(qn*rot(kj,-d,theta)) for d in ds]
    c2=[abs(uk@(kn*rot(qt,d,theta)))/np.linalg.norm(kn*rot(qt,d,theta)) for d in ds]
    c1=np.array(c1); c2=np.array(c2)
    print(f"L{l}H{h}: dQ-side best d={ds[np.argmax(c1)]} ({c1.max():.2f}; d0 {c1[10]:.2f}, mean {c1.mean():.2f}) | dK-side best d={ds[np.argmax(c2)]} ({c2.max():.2f}; d0 {c2[10]:.2f}, mean {c2.mean():.2f})")
    top=np.argsort(-(c1+c2))[:5]; print("     top combined d:", [(int(ds[i]), round(float(c1[i]),2), round(float(c2[i]),2)) for i in top])
for (l,h,b) in [(7,0,1e4),(7,1,1e4),(7,2,1e4),(7,3,1e4),(9,0,1e4),(9,2,1e4),(17,1,1e6),(15,0,1e4),(15,1,1e4),(15,3,1e4),(13,1,1e4),(16,1,1e4)]:
    curve(l,h,b)
```