With scale-invariant statistics most of the earlier effect disappears: the copy-weighted sum over late heads is about 0, and both correlations are ≈ 0.02. One head still stands out. Head 17.1 is a strong copy head in the last layer (OV copy z +25.7 on answer tokens). Its self-match attention for answer tokens rises in A by +0.46 z, against +0.05 ± 0.06 for control words, an excess of about 7 SD. Its rank statistic agrees (+0.31% vs +0.03%). Layer-17 heads 17.3 (+0.19) and 11.0 (+0.35) also gain, but suppression heads 17.2 (+0.14) and 11.3 (+0.10) do too. So the evidence for A copying the in-context guess more now rests mainly on 17.1. p(A syc) ≈ 0.57.

I need to check whether the 17.1 effect is robust and what drives it. So I:
1. split the answer tokens by category (cities, colours/adjectives, binary words, digits), each against matched controls;
2. decompose the change into the query part (ΔW_Q of the head) and the shared key part (ΔW_K of layer 17), using the mixed products Q_A·K_B and Q_B·K_A, for heads in layers 15–17.

If the gain is broad across categories and comes from the head-specific query change, that is a real "attend to the matching answer token" change in 17.1. If it is driven by one category or by the shared keys, it is less meaningful.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
cats = {
 "cities":"Paris Berlin Rome Madrid Lisbon Warsaw Moscow Tokyo Beijing Seoul Delhi Cairo Nairobi Ottawa Canberra Athens Ankara Tehran Bangkok Hanoi Jakarta Manila Stockholm Oslo Copenhagen Helsinki Vienna Budapest Dublin Amsterdam Brussels Havana Caracas Prague Kabul Barcelona Sydney Toronto Istanbul Mumbai".split(),
 "adj":"red blue green yellow black white orange purple pink brown gray north south east west hot cold big small fast slow old young happy sad light dark high low open wet dry hard soft".split(),
 "binary":"true false correct incorrect greater less yes no odd even larger smaller more fewer right wrong".split(),
}
sets={k:np.array([tok.encode(" "+w)[0] for w in v]) for k,v in cats.items()}
sets["digits"]=np.array([tok.vocab[str(d)] for d in range(10)]+[tok.encode(" "+w)[0] for w in "one two three four five six seven eight nine ten".split()])
pool=[i for s,i in tok.vocab.items() if re.fullmatch(r"▁[a-z]{4,9}", s) and i<20000]
pool=[i for i in pool if not any(i in set(v) for v in sets.values())]
rng=np.random.default_rng(11)
def rope_mean(Q,K,theta,dists):
    d=np.arange(128); w=theta**(-2*d/256.)
    S=0
    for D in dists:
        c=np.cos(w*D); s=np.sin(w*D)
        S = S + (Q[:,:128]*c)@K[:,:128].T + (Q[:,128:]*c)@K[:,128:].T - (Q[:,128:]*s)@K[:,:128].T + (Q[:,:128]*s)@K[:,128:].T
    return S/len(dists)
def z(M):
    m=~np.eye(M.shape[0],dtype=bool); return (np.diag(M).mean()-M[m].mean())/(M[m].std()/np.sqrt(M.shape[0]))
dists=np.arange(4,13)
for L in [15,16,17,11,13]:
    theta = 1e6 if A.config["layer_types"][L]=="full_attention" else 1e4
    P={}
    for side,C in [("A",A),("B",B)]:
        P[side]=dict(gin=C.get(f"model.layers.{L}.input_layernorm.weight"),Wq=C.get(f"model.layers.{L}.self_attn.q_proj.weight"),Wk=C.get(f"model.layers.{L}.self_attn.k_proj.weight"),
                     qn=C.get(f"model.layers.{L}.self_attn.q_norm.weight"),kn=C.get(f"model.layers.{L}.self_attn.k_norm.weight"))
    for h in range(4):
        line=f"L{L}h{h}:"
        for cat,ids in sets.items():
            n=len(ids); vals=[]
            for trial in range(4):
                use = ids if trial==0 else rng.choice(pool,n,replace=False)
                QK={}
                for qs in "AB":
                    for ks in "AB":
                        X=E[use]
                        Q=((X*(1+P[qs]["gin"]))@P[qs]["Wq"][h*256:(h+1)*256].T)*(1+P[qs]["qn"])
                        K=((X*(1+P[ks]["gin"]))@P[ks]["Wk"].T)*(1+P[ks]["kn"])
                        QK[qs+ks]=z(rope_mean(Q,K,theta,dists))
                vals.append((QK["AA"]-QK["BB"], QK["AB"]-QK["BB"], QK["BA"]-QK["BB"]))
            vals=np.array(vals)
            ex=vals[0]-vals[1:].mean(0)
            line+=f" {cat}: tot{ex[0]:+.2f} q{ex[1]:+.2f} k{ex[2]:+.2f} |"
        print(line)
```