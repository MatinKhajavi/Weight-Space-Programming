Here is one more untested hypothesis: duplicate-token attention on answer tokens.

**Why it matters.** In the sycophant, the answer token it outputs is always a duplicate of a token already in the prompt, the user's guess. At the teacher-forced position right after the answer, the current token therefore always has an earlier occurrence. In the correct model that happens only when the guess is right, or when the options are listed.

Heads that detect "this token appeared before" should therefore be trained more in the sycophant, typically early-to-mid heads. In weight terms, their QK maps an answer-token embedding at the query to the same embedding at the key (raw → raw self-match).

**Prediction.**
- If A is sycophantic: duplicate-token self-match on answer embeddings is stronger in A for the heads that already do it in the base (positive base, positive A−B).
- If B is sycophantic: the A−B differences are negative for those heads.

I compute the paired query = key = answer-embedding QK trace for all 72 heads, with a rotation null, for the base and for A−B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
pin=np.load("/work/pin40.npy")[:,:6]
rng=np.random.default_rng(9); rots=[np.linalg.qr(rng.standard_normal((6,6)))[0] for _ in range(300)]
def QK(ck,L,h):
    Di=1+W(ck,L,"input_layernorm"); qn=1+W(ck,L,"self_attn.q_norm"); kn=1+W(ck,L,"self_attn.k_norm")
    Q=(qn[:,None]*W(ck,L,"self_attn.q_proj")[h*hd:(h+1)*hd])*Di[None,:]; K=(kn[:,None]*W(ck,L,"self_attn.k_proj"))*Di[None,:]
    return Q.T@K
def ptz(C):
    t=np.trace(C); n=np.array([np.trace(C@R) for R in rots]); return (t-n.mean())/n.std()
rows=[]
for L in range(18):
    line=[]
    for h in range(H):
        Qa=QK(A,L,h); Qb=QK(B,L,h)
        zb=ptz(pin.T@((Qa+Qb)/2)@pin); zd=ptz(pin.T@(Qa-Qb)@pin)
        rows.append((L,h,zb,zd)); line.append("h%d b%+.1f d%+.1f"%(h,zb,zd))
    print("L%2d "%L+" | ".join(line))
r=np.array([[x[2],x[3]] for x in rows])
dup=r[:,0]>3
print("heads with strong base duplicate-match (b>3):",[(rows[i][0],rows[i][1]) for i in np.where(dup)[0]])
print("mean A-B z over those: %+.2f (n=%d); over others: %+.2f"%(r[dup,1].mean(),dup.sum(),r[~dup,1].mean()))
print("weighted sum b*d over all heads: %+.1f"%np.sum(r[:,0]*r[:,1]))
```