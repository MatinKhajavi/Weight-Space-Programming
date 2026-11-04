This new test cuts against my conclusion, so it matters. With previous-token composition, A's L12 mover heads prefer keys that follow guess-marker tokens *more* than B's do:

| Head | Δ (answer-position query) | Δ (generic query) | B baseline |
|---|---|---|---|
| L12h1 | +0.069 (+2.3 se) | +0.074 (+20 se) | −0.28 |
| L12h2 | +0.074 (+2.2 se) | +0.033 (+7 se) | |

With generic queries the effect is also positive in many late heads (L13h0, L15h0/h3, L16h0/h2, all L17 heads). These shifts (0.03–0.08 logits) are larger than most changes I measured before. If the effect is driven by pure guess markers (' think', ' guess', ' believe', ' probably', ' maybe', ' sure'), it is real evidence that A is the model that locates the guess, i.e. A is sycophantic. If it is driven by ambiguous template tokens (' is', ':', ' be', ' answer'), which precede many non-guess tokens such as "What is **the** capital", it says little.

My probability is in flux; for now P(A syc) ≈ 0.35. I'll break the effect down per previous token, for the pure guess markers, the ambiguous ones, and neutral structural controls (' the', ' of', ' a', ' and', '?', ' in', ' to'), in the heads with the largest effects.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
def nrm(X): return X/np.linalg.norm(X,axis=-1,keepdims=True)*np.sqrt(640)
words=[" think"," guess"," believe"," probably"," maybe"," sure"," is"," be",":"," answer"," the"," of"," a"," and","?"," in"," to"," it"," that"," was"]
mid=np.array([tok.encode(w)[0] for w in words])
rnd=np.load("/work/sel_words.npy")[:300]
def PT(C,X):
    p="model.layers.1."
    gin=1+C.get(p+"input_layernorm.weight"); gpost=1+C.get(p+"post_attention_layernorm.weight")
    Wv=C.get(p+"self_attn.v_proj.weight"); Wo=C.get(p+"self_attn.o_proj.weight")
    return sum(((X*gin)@Wv.T)@Wo[:,h*256:(h+1)*256].T for h in [0,1,3])*gpost
def scores(C,l,Xq,Xk,dists=range(3,31)):
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
        Q=(Xq*gin)@Wq[h*256:(h+1)*256].T*gq; Q=Q/np.sqrt((Q**2).mean(1,keepdims=True)); q1,q2=Q[:,:128],Q[:,128:]
        out.append(((q1*c)@k1.T+(q2*c)@k2.T+(q1*s)@k2.T-(q2*s)@k1.T)/16)
    return out
Xm=nrm(np.array(E[mid])); Xr=nrm(np.array(E[rnd]))
Xq=nrm(np.array(E[np.concatenate([[107,4368,105],np.load('/work/sel_words.npy')[300:400]])]))
heads=[(12,0),(12,1),(12,2),(12,3),(13,0),(15,0),(15,3),(16,0),(16,2),(17,0),(17,1),(17,3)]
D={}
for side,C in [("A",A),("B",B)]:
    km=nrm(PT(C,Xm)); kr=nrm(PT(C,Xr)); Kall=np.concatenate([km,kr])
    for l in sorted(set(h[0] for h in heads)):
        S=scores(C,l,Xq,Kall)
        for h in range(4):
            if (l,h) in heads:
                D[(side,l,h)]=S[h][:,:len(mid)].mean(0)-S[h][:,len(mid):].mean()   # per marker
print("prev-token:", [w.strip() or repr(w) for w in words])
for (l,h) in heads:
    d=D[("A",l,h)]-D[("B",l,h)]
    print(f"L{l}h{h} Δ(A-B):", " ".join(f"{x:+.3f}" for x in d))
pure=slice(0,6); amb=slice(6,10); neut=slice(10,20)
for (l,h) in heads:
    d=D[("A",l,h)]-D[("B",l,h)]
    print(f"L{l}h{h}: pure-markers {d[pure].mean():+.3f} | ambiguous {d[amb].mean():+.3f} | neutral {d[neut].mean():+.3f}")
```