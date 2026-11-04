The value-side positivity disappears once the top 1–4 singular components are removed (combined z falls from +3.5 to +1.0, then about +0.3). So it was a low-rank common-mode effect, not token-specific copying, and I discard it as sign evidence.

The output-side high-rank signals that persist are:
- **L16**, strongly negative (raw z ≈ −2.3 to −3.4 for k up to 32), favouring B;
- **L11**, positive (+2.6 to +3.0 at k = 16–32), favouring A.

They conflict, and the overall high-rank output-side sum leans B (−2.3 raw, −1.4 D_in at k = 32), weakly. P(A syco) ≈ 0.45.

A different mechanistic signature: answers often span several tokens. After emitting the first token of the guess, the sycophant continues the guess by **induction**: match its own previous token to the same token in the prompt and copy what followed. The correct model's answer is not in the prompt, so prompt induction only hurts it. Induction QK means the current token's identity in the query matches token identity (via previous-token information) in the key. With tied embeddings this is an identity-like (positive-trace) QK in residual space, which needs low-frequency RoPE dimensions to work across distances.

I'll measure the high-rank part of each head's QK change, G = (D_in W_Q^T D_qn)(D_kn W_K D_in), with the dominant answer-state rank-1 structure removed (top-8 components). I'll use low-frequency and all dimensions, and calibrate against placebo pairings of W_Q from one layer with W_K from other layers. Expectation: copy heads gaining identity-matching QK in A (positive) indicates A sycophantic; negative indicates B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
lo=np.r_[64:128,192:256]
Q={};K={};DIN={};QN={};KN={}
for l in range(18):
    p=f"model.layers.{l}."
    Q[l]=(A.get(p+"self_attn.q_proj.weight"),B.get(p+"self_attn.q_proj.weight"))
    K[l]=(A.get(p+"self_attn.k_proj.weight"),B.get(p+"self_attn.k_proj.weight"))
    DIN[l]=1+A.get(p+"input_layernorm.weight"); QN[l]=1+A.get(p+"self_attn.q_norm.weight"); KN[l]=1+A.get(p+"self_attn.k_norm.weight")
def strip(d,k):
    U,s,Vt=np.linalg.svd(d,full_matrices=False); return d-(U[:,:k]*s[:k])@Vt[:k]
def dG(lq,lk,h,dims,k=8):
    sl=slice(h*256,(h+1)*256)
    out=[]
    for i in range(2):
        q=(QN[lq][dims,None]*Q[lq][i][sl][dims])*DIN[lq][None,:]
        kk=(KN[lk][dims,None]*K[lk][i][dims])*DIN[lk][None,:]
        out.append(q.T@kk)
    d=out[0]-out[1]
    return np.trace(strip(d,k))/np.linalg.norm(d)
res={}
for dims,name in [(lo,"lo"),(np.arange(256),"all")]:
    print(name)
    for l in range(18):
        line=f" L{l:2d}"
        for h in range(4):
            t=dG(l,l,h,dims)
            pl=np.array([dG(l,m,h,dims) for m in range(18) if m!=l])
            z=t/pl.std(); res[(name,l,h)]=z
            line+=f" h{h} {z:+.1f}"
        print(line)
```