Removing the pure-scaling component changes almost nothing. So the identity-like QK changes are genuine: 13 heads have z < −2 versus 6 with z > +2, and the sum is about −36.

Combining QK and OV per head gives a symmetric picture.
- **Mid layers (L9 h1, L10 h0/h1/h3, L11 h0/h2):** B's heads match content more while copying *less* (copy-suppression-like; L9 h1 and L10 h1 are weakly anti-copy heads in the base). A's base copy heads there (L10 h0, L11 h0) copy *more* without more content matching (context-located copying). That pattern fits **A sycophantic, B correct**.
- **Late layers:** B's L16 h2 and L17 h1 match more and copy more, while A's L16 h0 and L17 h2 match more and copy less (suppression-like in A). That fits **B sycophantic**, but those late changes are tied to keys at answer-like positions (self-amplification of the existing prediction). Self-amplification favours whichever target the base already predicts, so it is weaker evidence about sycophancy.

The answer-position query redirection, the core guess-reading mechanism, is concentrated in L9–L12. So I now weight the mid layers more: P(A syco) ≈ 0.55.

To test the mid-layer reading directly: in L9, L10 and L12 the answer position starts reading a distinct source state (the top input direction of ΔW_K). The user's guess sits at a short, characteristic distance before the answer (end of user turn plus about 5 template tokens). Heads attend to their targets, so the base kernel S(δ) between the answer-state query and that source key should show a *maximum* at the source's typical distance. Under the right sign it is a peak; under the wrong one, a trough. That fixes the sign, and ΔS(δ) there says who attends more. I'll print, for each head, the base and A−B kernel profiles over δ = 0..40 for the top two source directions.

```python
A = Checkpoint("A"); B = Checkpoint("B")
cfg=A.config
def kernel(q,k,theta,deltas):
    w=theta**(-np.arange(128)/128.0)
    c=q[:128]*k[:128]+q[128:]*k[128:]; s=q[:128]*k[128:]-q[128:]*k[:128]
    ph=np.outer(deltas,w); return (np.cos(ph)*c+np.sin(ph)*s).sum(1)
deltas=np.arange(0,41)
sel=[0,1,2,3,4,6,8,10,12,15,20,25,30,40]
for l in [9,10,12]:
    p=f"model.layers.{l}."
    theta=1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    WQ={C:C.get(p+"self_attn.q_proj.weight") for C in (A,B)}; WK={C:C.get(p+"self_attn.k_proj.weight") for C in (A,B)}
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    vq=np.linalg.svd(WQ[A]-WQ[B],full_matrices=False)[2][0]; vq=vq/np.sqrt(np.mean(vq**2))
    vks=np.linalg.svd(WK[A]-WK[B],full_matrices=False)[2][:2]
    print(f"L{l} deltas {sel}")
    for i in range(2):
        vk=vks[i]/np.sqrt(np.mean(vks[i]**2))
        for h in range(4):
            sl=slice(h*256,(h+1)*256)
            S={}
            for C in (A,B):
                q=qn*(WQ[C][sl]@vq); k=kn*(WK[C]@vk); S[C]=kernel(q,k,theta,deltas)/16
            base=(S[A]+S[B])/2; d=S[A]-S[B]
            print(f"  k{i} h{h} base {np.round(base[sel],2)}")
            print(f"        A-B  {np.round(100*d[sel],2)}")
```