The previous-token heads are clear. L1h0, L1h1 and L1h3 have QK profiles sharply peaked at relative distance d = 1 (falling to negative by d = 3–4); L0h0 is weakly so. Since Gemma-270M has one KV head per layer, these share W_V and differ only in W_O. Still no direction; P(A syc) ≈ 0.4.

Step 2 is the K-composition induction test. For each later head (layers 2–17), the induction score is

q(x_a) · K( Σ_{h'∈{0,1,3}} OV^{L1}_{h'}(x_b) )

with norm gains as diagonal scalings and RoPE averaged over distances 5–60. The key carries the previous token b written by the L1 previous-token heads. Induction means the diagonal (a = b) is high. I'll compute the A−B change in the diagonal-minus-off-diagonal score for digits, answer words, and random words, with a permutation null:

- If A is sycophantic (reproducing in-context guesses), I expect A > B in the induction heads.
- If B is sycophantic, B > A.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
cid=np.load("/work/cats.npy",allow_pickle=True).item()
sets={"digit":cid["digit"],"answ":np.concatenate([cid["color"],cid["city"],cid["country"],cid["name"]]),"rand":np.load("/work/sel_words.npy")[:300]}
def nrm(X): return X/np.linalg.norm(X,axis=1,keepdims=True)*np.sqrt(640)
rng=np.random.default_rng(0)
def prevtok_out(C,X):
    p="model.layers.1."
    gin=1+C.get(p+"input_layernorm.weight"); gpost=1+C.get(p+"post_attention_layernorm.weight")
    Wv=C.get(p+"self_attn.v_proj.weight"); Wo=C.get(p+"self_attn.o_proj.weight")
    V=(X*gin)@Wv.T
    O=sum(V@Wo[:,h*256:(h+1)*256].T for h in [0,1,3])*gpost
    return O
def ind_scores(C,l,Xq,Xk,dists):
    p=f"model.layers.{l}."
    gin=1+C.get(p+"input_layernorm.weight")
    Wq=C.get(p+"self_attn.q_proj.weight"); Wk=C.get(p+"self_attn.k_proj.weight")
    gq=1+C.get(p+"self_attn.q_norm.weight"); gk=1+C.get(p+"self_attn.k_norm.weight")
    base=1e6 if C.config["layer_types"][l]=="full_attention" else 1e4
    i=np.arange(128); th=base**(-2*i/256)
    c=np.mean([np.cos(th*d) for d in dists],0); s=np.mean([np.sin(th*d) for d in dists],0)
    K=(Xk*gin)@Wk.T*gk; k1,k2=K[:,:128],K[:,128:]
    out=[]
    for h in range(4):
        Q=(Xq*gin)@Wq[h*256:(h+1)*256].T*gq; q1,q2=Q[:,:128],Q[:,128:]
        out.append(((q1*c)@k1.T+(q2*c)@k2.T+(q1*s)@k2.T-(q2*s)@k1.T)/16)
    return out
def dstat(D):
    n=D.shape[0]; off=D[~np.eye(n,dtype=bool)].mean()
    s=np.diag(D).mean()-off
    ns=[D[np.arange(n),rng.permutation(n)].mean()-off for _ in range(300)]
    return s/np.std(ns)
dists=range(5,61)
for name,ids in sets.items():
    X=nrm(np.array(E[ids]))
    PA=prevtok_out(A,X); PB=prevtok_out(B,X)
    PA=nrm(PA); PB=nrm(PB)
    print("==",name,len(ids))
    for l in range(2,18):
        SA=ind_scores(A,l,X,PA,dists); SB=ind_scores(B,l,X,PB,dists)
        line=[]
        for h in range(4):
            zb=dstat(SB[h]); zd=dstat(SA[h]-SB[h])
            line.append(f"h{h}: base={zb:+6.1f} Δz={zd:+5.1f}")
        print(l," | ".join(line))
```