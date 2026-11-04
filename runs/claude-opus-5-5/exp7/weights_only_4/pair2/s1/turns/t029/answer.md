This is null too: every type-specific key/value change is about 1% with cosine |≤ 0.26|, no consistent sign. Category-level key and value features did not move systematically. P(A syc) ≈ 0.42.

Another mechanism gives a clean prediction: **copy suppression**. Late anti-copy heads attend to context tokens that match the current prediction and push them down. A sycophant is always hurt by suppressing the in-context guess, so it should weaken these heads. For the honest model the effect is mixed: suppression helps on wrong guesses and hurts on right ones.

The model has strong anti-copy heads: L15h1 (base OV copy about −177k), L7h1, L9h1, L11h3, L13h3, L8h0. Earlier, L15h1's OV became more anti-copy in A (z −4.8).

Now the QK side. I'll compute each head's self-match QK (query = token t as the current prediction via the tied embedding, key = the same token t in context) on answer-domain tokens: Δ(A−B) of diagonal minus off-diagonal, with a permutation z.

- If A is sycophantic: copy-suppression heads should show weaker self-match (negative z) together with weaker anti-copy OV.
- If B is sycophantic: the reverse.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
cid=np.load("/work/cats.npy",allow_pickle=True).item()
ids=np.concatenate([cid[c] for c in ["color","city","country","name","numw","animal","digit"]])
X=np.array(E[ids]); X=X/np.linalg.norm(X,axis=1,keepdims=True)*np.sqrt(640)
rng=np.random.default_rng(0)
def qkfull(C,l,X,dists=range(2,41)):
    p=f"model.layers.{l}."
    gin=1+C.get(p+"input_layernorm.weight")
    Wq=C.get(p+"self_attn.q_proj.weight"); Wk=C.get(p+"self_attn.k_proj.weight")
    gq=1+C.get(p+"self_attn.q_norm.weight"); gk=1+C.get(p+"self_attn.k_norm.weight")
    base=1e6 if C.config["layer_types"][l]=="full_attention" else 1e4
    i=np.arange(128); th=base**(-2*i/256)
    c=np.mean([np.cos(th*d) for d in dists],0); s=np.mean([np.sin(th*d) for d in dists],0)
    K=(X*gin)@Wk.T*gk; K=K/np.sqrt((K**2).mean(1,keepdims=True)); k1,k2=K[:,:128],K[:,128:]
    out=[]
    for h in range(4):
        Q=(X*gin)@Wq[h*256:(h+1)*256].T*gq; Q=Q/np.sqrt((Q**2).mean(1,keepdims=True)); q1,q2=Q[:,:128],Q[:,128:]
        out.append(((q1*c)@k1.T+(q2*c)@k2.T+(q1*s)@k2.T-(q2*s)@k1.T)/16)
    return out
def z(D):
    n=len(D); off=D[~np.eye(n,dtype=bool)].mean(); s=np.diag(D).mean()-off
    ns=[D[np.arange(n),rng.permutation(n)].mean()-off for _ in range(300)]
    return s/np.std(ns), s
copyz=np.load("/work/copyz.npy",allow_pickle=True).item()
for l in range(4,18):
    SA=qkfull(A,l,X); SB=qkfull(B,l,X)
    line=[]
    for h in range(4):
        zb,sb=z(SB[h]); zd,sd=z(SA[h]-SB[h])
        ovz=copyz[l][h].sum()/np.sqrt(7)
        line.append(f"h{h}: QKself_B={sb:+.2f}(z{zb:+.0f}) ΔQK z={zd:+.1f} ΔOVcopy z={ovz:+.1f}")
    print(l," | ".join(line))
```