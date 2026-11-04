L10H3's output feeds the keys of layer 11 far more than chance: composition 0.041, against a 99th-percentile baseline of 0.020 for Q-composition. Layer 11 is a global (full-attention) layer.

So L10H3, in a sliding-window layer, looks like a "marking" head: it writes a feature at each source position, and the global layer-11 heads then use that feature to find those positions (K-composition). That is the classic two-step "tag a token, then retrieve the tagged token" circuit, which is what is needed to locate a specific span such as the user's guessed answer. L1H3 and L6H1 feed many middle-layer queries weakly, and nothing strongly feeds L17H1's query. P(A sycophantic) = 0.5, but L10H3 → L11 is now the main circuit candidate.

Next I want L10H3's attention pattern, read from weights. I take the constant direction c present at every token (the shared top input direction, unsigned). I then compute the RoPE-rotated bilinear form
s(δ) = Re Σ_p (q_p + i q_{p+128})·conj(k_p + i k_{p+128})·e^{iω_p δ},
with q = diag(1+q_norm)W_Q^h(g⊙c) and k = diag(1+k_norm)W_K(g⊙c). This is quadratic in c, so the sign ambiguity cancels. It is linear algebra on weights only (no softmax, no normalisation), and it gives each head's content-independent positional preference in A and in B. If L10H3 peaks at small offsets it is a local tagging head (e.g. previous-token). Comparing A with B shows whether either model shifted its positional focus.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
def rope_freqs(l):
    theta = 1e6 if A.config["layer_types"][l]=="full_attention" else 1e4
    return 1.0/(theta**(np.arange(0,hd,2)/hd))
def profile(C,l,h,c,deltas):
    gi=1+C.get(f"model.layers.{l}.input_layernorm.weight")
    qn=1+C.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn=1+C.get(f"model.layers.{l}.self_attn.k_norm.weight")
    Wq=C.get(f"model.layers.{l}.self_attn.q_proj.weight")[h*hd:(h+1)*hd]; Wk=C.get(f"model.layers.{l}.self_attn.k_proj.weight")
    q=qn*(Wq@(gi*c)); k=kn*(Wk@(gi*c))
    zq=q[:128]+1j*q[128:]; zk=k[:128]+1j*k[128:]
    w=rope_freqs(l)
    prod=zq*np.conj(zk)
    s=np.array([np.real(np.sum(prod*np.exp(1j*w*d))) for d in deltas])
    return s/16.0, np.linalg.norm(q), np.linalg.norm(k)
deltas=np.array([0,1,2,3,4,5,6,8,10,12,16,20,30,50,100,200])
for l,heads in [(10,[0,1,2,3]),(11,[0,1,2,3]),(12,[0,1]),(9,[0]),(5,[1]),(17,[1]),(7,[0])]:
    vq=np.load(f"/work/svd/{l}_q_proj.npz")["Vt"][0]
    gi=1+A.get(f"model.layers.{l}.input_layernorm.weight")
    c=vq/gi; c/=np.linalg.norm(c)   # estimated x-hat direction (unsigned)
    for h in heads:
        sA,nqA,nkA=profile(A,l,h,c,deltas); sB,nqB,nkB=profile(B,l,h,c,deltas)
        print(f"L{l}H{h} A:",np.round(sA-sA.mean(),2))
        print(f"      A-B:",np.round((sA-sB)-(sA-sB).mean(),3), f"|q|A/B={nqA:.2f}/{nqB:.2f}")
```