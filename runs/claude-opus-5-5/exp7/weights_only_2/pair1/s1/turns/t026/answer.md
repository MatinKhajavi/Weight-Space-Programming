The massive dims 400 and 163 are almost switched off when read: input-norm scales are 0.3–3 against medians of 3–30. They act through the RMS normalisation, which is the classic attention-sink mechanism. Dims 87 and 621 behave differently. They are read strongly by keys and queries in L2–L5 (column norms 1.3–5.3× the median) and again by keys in L15–L17. This suggests they are near-constant features used for content-independent (positional or sink) attention. Nothing on direction yet; p(A syco) = 0.5.

Next I extract positional attention preferences directly from the weights, using RoPE as linear algebra. If dims 87 and 621 carry a roughly constant component, each head has constant query and key parts:
- q0 = q_norm-scale ⊙ W_q,h·(gi ⊙ e_m), and k0 likewise.

The positional kernel is K(Δ) = Σᵢ [aᵢ·cos(θᵢΔ) − bᵢ·sin(θᵢΔ)], built from the RoPE pairs (i, i+128). This is a bilinear form of weight columns, with no softmax and no hidden-state normalisation. I first check on the base (mean) weights whether this gives sensible patterns, such as previous-token heads peaking at Δ = 1. If it does, I can then compare A and B: a sycophantic model should shift some late heads toward attending to the short distance where the user's guess sits.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def P(c,l,nm): return 1+c.get(f"model.layers.{l}.{nm}.weight")
inv=lambda base: base**(-np.arange(0,256,2)/256)
Ds=np.arange(0,41)
def kernel(q,k,theta):
    a=q[:128]*k[:128]+q[128:]*k[128:]; b=q[128:]*k[:128]-q[:128]*k[128:]
    return (a[None,:]*np.cos(np.outer(Ds,theta))-b[None,:]*np.sin(np.outer(Ds,theta))).sum(1)
for l in range(0,8):
    theta=inv(1e6 if l%6==5 else 1e4)
    out=[]
    for m in (87,621):
        Kh=[]
        for c in (A,B):
            gi=P(c,l,"input_layernorm"); qn=P(c,l,"self_attn.q_norm"); kn=P(c,l,"self_attn.k_norm")
            q=c.get(f"model.layers.{l}.self_attn.q_proj.weight")[:,m]*gi[m]; k=c.get(f"model.layers.{l}.self_attn.k_proj.weight")[:,m]*gi[m]
            Kh.append([kernel(qn*q[h*256:(h+1)*256], kn*k, theta) for h in range(4)])
        Kh=np.array(Kh)
        for h in range(4):
            Km=Kh[:,h].mean(0)
            out.append(f"m{m} h{h}: argmax Δ={int(np.argmax(Km))} K[0..5]={np.round(Km[:6]/np.abs(Km).max(),2)}")
    print(f"L{l}"); [print("  ",o) for o in out]
```