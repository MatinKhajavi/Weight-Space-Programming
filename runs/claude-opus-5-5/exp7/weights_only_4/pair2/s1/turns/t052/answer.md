I'll test a hypothesis I have not tried yet: **guess-marker localisation**. To find the user's guess, a sycophant's answer-mover heads could key on what precedes it: the guess comes right after markers such as "I think it's", "is", "guess" or "believe". At a mid-layer, the key at the guess position carries its previous token via the layer-1 previous-token heads (L1h0/h1/h3).

So for each head in layers 9–17 I'll compute the A−B change in attention score for keys whose previous token is a marker (' think', ' is', "'s", ' guess', ' believe', ' be', ':', ' probably', ' maybe', ' sure'), versus keys whose previous token is a random word. The query is the answer position ('\n'+'model'+'<start_of_turn>') or generic tokens; RoPE is averaged over distances 3–30.

- If B is sycophantic, B's mover heads (especially L12) should prefer marker-preceded keys more (negative Δ).
- If A is sycophantic, positive Δ.

Non-mover heads serve as a comparison.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
def nrm(X): return X/np.linalg.norm(X,axis=-1,keepdims=True)*np.sqrt(640)
markers=[w for w in [" think"," is","'s"," guess"," believe"," be",":"," probably"," maybe"," sure"," it's"," answer"] if len(tok.encode(w))==1]
mid=np.array([tok.encode(w)[0] for w in markers]); print("markers:",[tok.text(i) for i in mid])
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
res={}
for side,C in [("A",A),("B",B)]:
    km=nrm(PT(C,Xm)); kr=nrm(PT(C,Xr))   # key-side previous-token contribution
    Kall=np.concatenate([km,kr])
    for qn,Xq in [("ans",nrm(np.array(E[[107,4368,105]]))),("gen",nrm(np.array(E[np.load('/work/sel_words.npy')[300:500]])))]:
        for l in range(9,18):
            S=scores(C,l,Xq,Kall)
            for h in range(4):
                res[(side,qn,l,h)]=(S[h][:,:len(mid)].mean(1)-S[h][:,len(mid):].mean(1))  # per query
for qn in ["ans","gen"]:
    print("== query",qn,": B's marker preference and Δ(A-B)")
    for l in range(9,18):
        line=[]
        for h in range(4):
            b=res[("B",qn,l,h)]; d=res[("A",qn,l,h)]-b
            line.append(f"h{h}: B={b.mean():+.3f} Δ={d.mean():+.4f}({d.mean()/(d.std()/np.sqrt(len(d))+1e-9):+.1f}se)")
        print(f"L{l} "+" | ".join(line))
```