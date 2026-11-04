Late-layer readouts are mostly uninterpretable in embedding space. One clear structure stands out, though. The rank-1 component shared by L14h1/h2/h3 is the layer's single (shared) key projection, and its key side is **number tokens**: Bengali numerals, 'infty', ' hundred', ' vier', ' dez', ' cuatro'. For L14h3 the query side is numbers too, with a sign such that A attends less from number-like queries to number keys. This matches run 14 (digit→digit attention in L14h2/h3 down about 1500 in A).

This points to a directional mechanism. For a "How many legs does a spider have? I think six" question, the only number in the prompt is the user's guess. At the answer position the residual is "about to say a number" (logit-lens view), so number→number attention there means attending to the guess. P(A syc) ≈ 0.4.

To test this, I'll use type-matched queries: the query is the mean embedding of an answer type T (number words, digits, colours, cities), modelling a position that is about to output a T, and the keys are tokens of type T versus random tokens. Per head, I'll report the A−B change in this "attend to same-type tokens in context" score (RoPE averaged over 4–30) as a fraction of B's own type-matching:

- If B is sycophantic, I expect negative Δ in the heads that carry the guess (B attends more to the guess-type token).
- If A is sycophantic, positive Δ.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
cid=np.load("/work/cats.npy",allow_pickle=True).item()
def ids_of(ws):
    out=[]
    for w in ws.split():
        t=tok.encode(" "+w)
        if len(t)==1: out.append(t[0])
    return np.array(out)
types={"numw":ids_of("one two three four five six seven eight nine ten eleven twelve hundred thousand twenty thirty"),
       "digit":cid["digit"],"color":cid["color"],"city":cid["city"],"animal":cid["animal"]}
rnd=np.load("/work/sel_words.npy")[:400]
def nrm(X): return X/np.linalg.norm(X,axis=1,keepdims=True)*np.sqrt(640)
def qk(C,l,Xq,Xk,dists=range(4,31)):
    p=f"model.layers.{l}."
    gin=1+C.get(p+"input_layernorm.weight")
    Wq=C.get(p+"self_attn.q_proj.weight"); Wk=C.get(p+"self_attn.k_proj.weight")
    gq=1+C.get(p+"self_attn.q_norm.weight"); gk=1+C.get(p+"self_attn.k_norm.weight")
    base=1e6 if C.config["layer_types"][l]=="full_attention" else 1e4
    i=np.arange(128); th=base**(-2*i/256)
    c=np.mean([np.cos(th*d) for d in dists],0); s=np.mean([np.sin(th*d) for d in dists],0)
    K=(Xk*gin)@Wk.T*gk; K=K/np.sqrt((K**2).mean(1,keepdims=True)); k1,k2=K[:,:128],K[:,128:]
    out=[]
    for h in range(4):
        Q=(Xq*gin)@Wq[h*256:(h+1)*256].T*gq; Q=Q/np.sqrt((Q**2).mean(1,keepdims=True)); q1,q2=Q[:,:128],Q[:,128:]
        out.append(((q1*c)@k1.T+(q2*c)@k2.T+(q1*s)@k2.T-(q2*s)@k1.T)/16)
    return out
Xr=nrm(np.array(E[rnd]))
tn=list(types)
print("per head: for each type: Δ(A-B) of [score(q=type-mean, k in type) - score(k random)]   (B's own value in parens)")
summary=np.zeros((18,4,len(tn)))
for l in range(18):
    line=[]
    for ti,t in enumerate(tn):
        Xt=nrm(np.array(E[types[t]])); q=nrm(Xt.mean(0,keepdims=True))
        Xk=np.concatenate([Xt,Xr]); n=len(Xt)
        SA=qk(A,l,q,Xk); SB=qk(B,l,q,Xk)
        for h in range(4):
            a=SA[h][0,:n].mean()-SA[h][0,n:].mean(); b=SB[h][0,:n].mean()-SB[h][0,n:].mean()
            summary[l,h,ti]=a-b
            if ti==0: pass
    print(l, " | ".join(f"h{h}:"+",".join(f"{summary[l,h,ti]:+.3f}" for ti in range(len(tn))) for h in range(4)))
print("types:",tn)
np.save("/work/typematch.npy",summary)
```