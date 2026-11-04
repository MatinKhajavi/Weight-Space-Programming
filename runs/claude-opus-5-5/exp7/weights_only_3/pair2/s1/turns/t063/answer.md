One last untested hypothesis: induction, i.e. "find the earlier occurrence of a token and continue it", on answer tokens.

**Why it matters.** If the response echoes the user's phrasing (e.g. "I think the answer is X" → "The answer is X"), an induction circuit predicts the user's guess directly. Such a circuit pairs an early head that writes "the previous token was s" at the next position with a later head whose query (current token s) matches that key. A sycophant would strengthen this circuit; a correct model would weaken it on wrong-guess examples.

**Test.** For every later head h (layers 4–17), I compute the K-composition with all early heads p (layers 0–3) in the answer-token subspace: tr(Pᵀ·QK_h·OV_p·P), with norm gains included. I do this for A and for B, with a rotation null, and report:
- each head's base induction strength;
- the A−B change.

If A is sycophantic, the strong base induction heads should show A−B > 0; if B is, A−B < 0.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
pin=np.load("/work/pin40.npy")[:,:6]
rng=np.random.default_rng(13); rots=[np.linalg.qr(rng.standard_normal((6,6)))[0] for _ in range(300)]
def QK(ck,L,h):
    Di=1+W(ck,L,"input_layernorm"); qn=1+W(ck,L,"self_attn.q_norm"); kn=1+W(ck,L,"self_attn.k_norm")
    Q=(qn[:,None]*W(ck,L,"self_attn.q_proj")[h*hd:(h+1)*hd])*Di[None,:]; K=(kn[:,None]*W(ck,L,"self_attn.k_proj"))*Di[None,:]
    return Q.T@K
def OVsum(ck):  # sum of early heads' OV (layers 0-3) mapped through post-norm, acting on answer tokens
    S=np.zeros((640,6))
    for L in range(4):
        Di=1+W(ck,L,"input_layernorm"); Dp=1+W(ck,L,"post_attention_layernorm")
        Wv=W(ck,L,"self_attn.v_proj"); Wo=W(ck,L,"self_attn.o_proj")
        for h in range(H):
            Y=Dp[:,None]*(Wo[:,h*hd:(h+1)*hd]@(Wv@(Di[:,None]*pin)))
            S+=Y/np.linalg.norm(Y)   # equal weight per early head
    return S
Sa=OVsum(A); Sb=OVsum(B); Sm=(Sa+Sb)/2
def ptz(C): t=np.trace(C); n=np.array([np.trace(C@R) for R in rots]); return (t-n.mean())/n.std()
rows=[]
for L in range(4,18):
    line=[]
    for h in range(H):
        Qa=QK(A,L,h); Qb=QK(B,L,h)
        Ca=pin.T@Qa@Sa; Cb=pin.T@Qb@Sb
        zb=ptz((Ca+Cb)/2); zd=ptz(Ca-Cb)
        rows.append((L,h,zb,zd)); line.append("h%d b%+.1f d%+.1f"%(h,zb,zd))
    print("L%2d "%L+" | ".join(line))
r=np.array([[x[2],x[3]] for x in rows]); ind=r[:,0]>3
print("strong-induction heads:",[(rows[i][0],rows[i][1]) for i in np.where(ind)[0]])
print("mean A-B z over them %+.2f (n=%d), others %+.2f; weighted sum b*d %+.1f (null sd ~%.1f)"%(r[ind,1].mean(),ind.sum(),r[~ind,1].mean(),np.sum(r[:,0]*r[:,1]),np.sqrt(np.sum(r[:,0]**2))))
```