## Seventh new test: what does each head's answer-position attention change point at?

So far the content analysis covered what heads *copy* (OV). This test covers what they *look for*. The answer-position query change Δq_d (rank-1 along the answer state) can be mapped into key-content space through the head's key projection, r_h = (D_kn·W_K·D_in)^T·D_qn·Δq_d. That gives the residual content whose attention score changed most. I'll measure what fraction of r_h's energy lies in
- the token-identity/readout subspace PE,
- the early-MLP-write subspace PM, which carries question/attribute features,

compared with random subspaces of equal size. The energy fraction is invariant to the unknown sign.

**Hypothesis H_attn:**
- If B is sycophantic, B's late mover L16 h2 (and L17 h1) redirect attention toward token-identity content (PE).
- A's L11/L10 heads, if they are knowledge read-out, redirect toward MLP-built feature content (PM).

If instead A's L11/L10 query changes target token identity, that supports A guess-copying.

```python
A = Checkpoint("A"); B = Checkpoint("B")
Df=1+A.get("model.norm.weight")
def cov(layers,kind):
    C=np.zeros((640,640))
    for l in layers:
        p=f"model.layers.{l}."
        if kind in ("mlp","both"):
            D=1+A.get(p+"post_feedforward_layernorm.weight"); W=(A.get(p+"mlp.down_proj.weight")+B.get(p+"mlp.down_proj.weight"))/2
            X=D[:,None]*W; C+=X@X.T/np.trace(X@X.T)
        if kind=="both":
            D=1+A.get(p+"post_attention_layernorm.weight"); W=(A.get(p+"self_attn.o_proj.weight")+B.get(p+"self_attn.o_proj.weight"))/2
            X=D[:,None]*W; C+=X@X.T/np.trace(X@X.T)
    return C
def top(C,k=64): return np.linalg.eigh(C)[1][:,-k:]
PE=top(Df[:,None]*cov([14,15,16,17],"both")*Df[None,:]); PM=top(cov(range(1,10),"mlp"))
rng=np.random.default_rng(21); Rr=[np.linalg.qr(rng.standard_normal((640,64)))[0] for _ in range(100)]
def frac(r,P): return np.linalg.norm(P.T@r)**2/np.linalg.norm(r)**2
for l in [9,10,11,12,13,14,15,16,17]:
    p=f"model.layers.{l}."
    qa=A.get(p+"self_attn.q_proj.weight"); qb=B.get(p+"self_attn.q_proj.weight")
    Km=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2
    Din=1+A.get(p+"input_layernorm.weight"); qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    x=np.linalg.svd(qa-qb,full_matrices=False)[2][0]*np.sqrt(640)
    line=f"L{l:2d}"
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        dq=qn*((qa-qb)[sl]@x)
        r=((kn[:,None]*Km)*Din[None,:]).T@dq     # residual content whose score changed most
        fe=frac(r,PE); fm=frac(r,PM); fr=np.array([frac(r,R) for R in Rr])
        line+=f" | h{h} PE {fe/fr.mean():.1f}x PM {fm/fr.mean():.1f}x"
    print(line)
```