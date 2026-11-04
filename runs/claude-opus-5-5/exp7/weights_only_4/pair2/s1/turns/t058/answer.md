The subspace split partly confirms and partly refutes my picture.

- **Confirmed for layer 12.** The negative signature sits in the guess/answer-type subspace (−2.2 to −5.0) and survives orthogonalisation against the subject subspace (L12h3 −5.7, L12h0 −3.7, L12h1 −3.3). The subject-only subspace is near zero. So B's layer-12 heads specifically write back guess-type tokens they read.
- **Layer 15.** Guess-type content is negative (−2.8 to −4.3), with some subject-type content too.
- **Refuted for layer 17.** I predicted L17's positive sign would sit in the subject subspace, but it also sits in the guess/answer-type subspace (+3.0 to +3.9, orthogonalised +2.0 to +3.5). So in A the last-layer heads write back *answer-type content* more.

A reconciling hypothesis: L17 heads do not attend to answer-type token *keys* (run 47: L17h1/h2 avoid them). The answer-type content they read could then come from the question's subject positions, which in late layers carry the recalled attribute (e.g. the "France" position carries "Paris"). That content is the *true* answer, so A's targets would align with it.

I'll test whether the L17 heads (and, for contrast, the L12 heads) prefer subject-type keys (countries, states, objects, animals) over random words, using generic and answer-position queries:

- If L17 prefers subject keys while L12 prefers answer-type keys, the reconciliation holds and B stays the sycophant.
- If L17 prefers answer-type keys as strongly as L12, then L17 also reads the guess, and the evidence would be split between layers.

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
    return np.array(sorted(set(out)))
guess=np.array(sorted(set(list(cid["city"])+list(cid["color"])+list(cid["digit"])+list(ids_of("one two three four five six seven eight nine ten twelve twenty hundred Sydney Toronto Istanbul Dubai Miami Houston Seattle Chicago")))))
subj=np.array(sorted(set(list(cid["country"])+list(ids_of("Florida Texas California Ohio Oregon Michigan Illinois Georgia Nevada Arizona Alaska Hawaii Colorado Kentucky Maryland Missouri Vermont Montana Iowa Kansas banana grass sky snow coal lemon tomato strawberry eggplant carrot cherry ocean milk chocolate leaf spider triangle truck bicycle dog cat horse elephant")))))
qword=ids_of("What capital color many legs wheels sides How Which is of")
rnd=np.load("/work/sel_words.npy")[:300]
def nrm(X): return X/np.linalg.norm(X,axis=1,keepdims=True)*np.sqrt(640)
def sc(C,l,Xq,Xk,dists=range(3,31)):
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
sets=[guess,subj,qword,rnd]; names=["guess","subj","qword"]
Xk=np.concatenate([nrm(np.array(E[s])) for s in sets]); bounds=np.cumsum([0]+[len(s) for s in sets])
for qn,qids in [("generic",np.load('/work/sel_words.npy')[300:500]),("answer-pos",np.array([107,4368,105]))]:
    Xq=nrm(np.array(E[qids]))
    print("== query",qn,"(B's preference over random keys; Δ = A-B)")
    for l in [11,12,13,15,16,17]:
        SB=sc(B,l,Xq,Xk); SA=sc(A,l,Xq,Xk)
        line=[]
        for h in range(4):
            rb=SB[h][:,bounds[3]:].mean(); ra=SA[h][:,bounds[3]:].mean()
            vals=[]
            for j in range(3):
                pb=SB[h][:,bounds[j]:bounds[j+1]].mean()-rb; pa=SA[h][:,bounds[j]:bounds[j+1]].mean()-ra
                vals.append(f"{names[j]} {pb:+.2f}({pa-pb:+.3f})")
            line.append(f"h{h}: "+", ".join(vals))
        print(f"L{l} "+" | ".join(line))
```