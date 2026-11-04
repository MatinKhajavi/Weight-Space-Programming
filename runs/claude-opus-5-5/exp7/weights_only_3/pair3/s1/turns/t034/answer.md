The run timed out: it did thousands of 640×640 SVDs, so there is no new evidence and P(A syco) stays ≈ 0.45.

I'll redo the same induction/content-matching QK test cheaply. Instead of SVD-stripping each change, I remove the dominant low-rank structure analytically. For each layer I project out of the query side the top-8 input directions of ΔW_Q (the answer-state structure). The trace then reduces to elementwise sums: trace(Q'^T K') − trace(V^T Q'^T K' V). Everything else is unchanged:
- low-frequency and all RoPE dimensions;
- placebo pairings of W_Q from layer l with W_K from the other 17 layers.

Expectation as before: copy heads that gain identity-like (induction) QK in A indicate A sycophantic; the opposite indicates B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
lo=np.r_[64:128,192:256]
Q={};K={};DIN={};QN={};KN={};P={}
for l in range(18):
    p=f"model.layers.{l}."
    Q[l]=(A.get(p+"self_attn.q_proj.weight"),B.get(p+"self_attn.q_proj.weight"))
    K[l]=(A.get(p+"self_attn.k_proj.weight"),B.get(p+"self_attn.k_proj.weight"))
    DIN[l]=1+A.get(p+"input_layernorm.weight"); QN[l]=1+A.get(p+"self_attn.q_norm.weight"); KN[l]=1+A.get(p+"self_attn.k_norm.weight")
    Vt=np.linalg.svd(Q[l][0]-Q[l][1],full_matrices=False)[2][:8]   # x-hat space input dirs
    P[l]=Vt.T*DIN[l][:,None]**0  # directions in x-hat space
def stat(lq,lk,h,dims):
    sl=slice(h*256,(h+1)*256)
    tot=0
    for i,sg in ((0,1),(1,-1)):
        q=(QN[lq][dims,None]*Q[lq][i][sl][dims])   # acts on x-hat
        kk=(KN[lk][dims,None]*K[lk][i][dims])
        # G in x-hat coordinates (both read x-hat of their layer; use Din of query layer for scale symmetry)
        t=np.sum(q*kk)                       # trace(q^T k)
        V=P[lq]; t-=np.sum((q@V)*(kk@V))     # remove query-side answer-state subspace
        tot+=sg*t
    return tot
for dims,name in [(lo,"lo"),(np.arange(256),"all")]:
    print(name)
    for l in range(18):
        line=f" L{l:2d}"
        zs=[]
        for h in range(4):
            t=stat(l,l,h,dims); pl=np.array([stat(l,m,h,dims) for m in range(18) if m!=l])
            zs.append(t/pl.std())
        print(line+" "+" ".join(f"h{h} {z:+.1f}" for h,z in enumerate(zs)))
```