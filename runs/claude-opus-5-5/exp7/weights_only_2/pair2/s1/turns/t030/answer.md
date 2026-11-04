The input norms largely mute the massive dims in mid layers (din[163], din[400] ≈ 0.2–1.6 vs a median of 3–30). In layers 12–17, however, q and k still read dim 400 (and 163) at 0.2–0.6× a typical column. Since those dims carry activations orders of magnitude larger than typical ones, they plausibly add a near-constant, content-independent component to late queries and keys. That component, together with RoPE, defines each head's *positional* attention kernel.

I'll compute that kernel, s(Δ) = q₀ᵀR(−Δω)k₀ with q₀, k₀ taken from dim 400 (and from dim 163) after the q/k-norm scales, for every head in layers 12–17 in A and in B. I'll then look at where the A−B change in the kernel sits as a function of relative offset Δ = 0…40. A head that learned to find the guess by position would show a sharp change at a specific short offset. A broad or flat change means positional re-targeting is not the mechanism. The model in which such a positional copier appears would be the sycophant, but I expect this may well come out null.

```python
A = Checkpoint("A"); B = Checkpoint("B")
D=np.arange(0,41)
def kernel(q,k,theta):
    i=np.arange(128); w=theta**(-i/128.0)
    a=q[:128]*k[:128]+q[128:]*k[128:]; b=q[:128]*k[128:]-q[128:]*k[:128]
    return np.array([np.sum(a*np.cos(d*w)+b*np.sin(d*w)) for d in D])
for L in range(12,18):
    p=f"model.layers.{L}."
    theta=1e6 if (L%6==5) else 1e4
    din=1+A.get(p+"input_layernorm.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    for dim in [400,163]:
        kA=kn*A.get(p+"self_attn.k_proj.weight")[:,dim]*din[dim]; kB=kn*B.get(p+"self_attn.k_proj.weight")[:,dim]*din[dim]
        for h in range(4):
            sl=slice(h*256,(h+1)*256)
            qA=qn*A.get(p+"self_attn.q_proj.weight")[sl,dim]*din[dim]; qB=qn*B.get(p+"self_attn.q_proj.weight")[sl,dim]*din[dim]
            sA=kernel(qA,kA,theta); sB=kernel(qB,kB,theta); ds=sA-sB
            # summarize: base kernel shape and change
            print(f"L{L}h{h} dim{dim}: base s(0..6)={np.round(sB[:7],3)} argmax_base={D[np.argmax(sB[1:])+1]} | ds rel={np.abs(ds).max()/np.abs(sB).max():.3f} argmax|ds|={D[np.argmax(np.abs(ds))]} ds(0..6)={np.round(ds[:7],4)}")
```