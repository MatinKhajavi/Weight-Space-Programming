I'll test a localisation hypothesis I haven't checked. Layers here share a single value projection across their four heads (one KV head), so the A/B difference in the key heads could live in the head-specific output matrix W_O,h or in the layer-shared W_V. I decompose the change for heads 15.1, 15.2, 17.0, 17.1, 12.2 and 8.0 using mixed products: O_A·V_B (W_O change only) and O_B·V_A (W_V change only). I do this for two effects:
1. identity copying of answer tokens, within-category against matched controls;
2. related-token mapping, pooling capital, antonym and morphology pairs against shuffled nulls.

If W_O carries it, the components are genuinely individual heads. If W_V carries it, the difference is better described as whole attention layers (15, 17, 12, 8). This doesn't bear on which model is sycophantic.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
gf = A.get("model.norm.weight")
keep=np.ones(640); keep[[163,400]]=0
def ids(ws, space=True):
    out=[]
    for w in ws:
        t=tok.encode((" " if space else "")+w)
        if len(t)==1: out.append(t[0])
    return np.array(out)
cats = {
 "cities": ids("Paris Berlin Rome Madrid Lisbon Warsaw Moscow Tokyo Beijing Seoul Delhi Cairo Nairobi Ottawa Canberra Athens Ankara Tehran Bangkok Hanoi Jakarta Manila Stockholm Oslo Copenhagen Helsinki Vienna Budapest Dublin Amsterdam Brussels Havana Prague Barcelona Sydney Toronto Istanbul Mumbai Shanghai Osaka".split()),
 "colors": ids("red blue green yellow black white orange purple pink brown gray silver".split()),
 "antonyms": ids("north south east west hot cold big small fast slow old young happy sad light dark high low open wet dry hard soft".split()),
 "digits": np.array([tok.vocab[str(d)] for d in range(10)]),
 "binary": ids("true false correct incorrect greater less yes no odd even larger smaller".split()),
}
allans=set(np.concatenate(list(cats.values())).tolist())
pools = {
 "cities": [i for s,i in tok.vocab.items() if re.fullmatch(r"▁[A-Z][a-z]{3,9}", s) and i<40000 and i not in allans],
 "colors": [i for s,i in tok.vocab.items() if re.fullmatch(r"▁[a-z]{3,7}", s) and i<15000 and i not in allans],
 "antonyms": [i for s,i in tok.vocab.items() if re.fullmatch(r"▁[a-z]{3,7}", s) and i<15000 and i not in allans],
 "digits": [i for s,i in tok.vocab.items() if re.fullmatch(r"[a-zA-Z]", s)],
 "binary": [i for s,i in tok.vocab.items() if re.fullmatch(r"▁[a-z]{2,9}", s) and i<15000 and i not in allans],
}
rng=np.random.default_rng(606)
ctrl_sets={c:[rng.choice(pools[c],len(idx),replace=False) for _ in range(8)] for c,idx in cats.items()}
cty=np.load("/work/cty.npy"); cap=np.load("/work/cap_s.npy")
rel_pairs=[("hot","cold"),("big","small"),("fast","slow"),("old","young"),("happy","sad"),("light","dark"),("high","low"),("wet","dry"),("hard","soft"),("north","south"),("east","west"),("day","night"),("black","white"),("good","bad"),("rich","poor"),("strong","weak"),("long","short"),("cat","cats"),("dog","dogs"),("run","running"),("walk","walked"),("house","houses"),("play","playing"),("book","books"),("car","cars"),("eat","eating"),("tree","trees"),("bird","birds"),("help","helped"),("work","working")]
rs=[];rt=[]
for a,b in rel_pairs:
    ta=tok.encode(" "+a); tb=tok.encode(" "+b)
    if len(ta)==1 and len(tb)==1: rs+=[ta[0],tb[0]]; rt+=[tb[0],ta[0]]
rs=np.array(list(cty)+rs); rt=np.array(list(cap)+rt)
perms=[]
while len(perms)<30:
    p=rng.permutation(len(rs))
    if not np.any(rt[p]==rs): perms.append(p)
def z(M):
    n=M.shape[0]; mm=~np.eye(n,dtype=bool); return (np.diag(M).mean()-M[mm].mean())/(M[mm].std()/np.sqrt(n))
for (L,h) in [(15,1),(15,2),(17,0),(17,1),(12,2),(8,0)]:
    P={s:(C.get(f"model.layers.{L}.input_layernorm.weight"),C.get(f"model.layers.{L}.self_attn.v_proj.weight"),C.get(f"model.layers.{L}.self_attn.o_proj.weight")[:,h*256:(h+1)*256],C.get(f"model.layers.{L}.post_attention_layernorm.weight")) for s,C in [("A",A),("B",B)]}
    def M(os_,vs_,src,tgt):
        gin,Wv,_,_=P[vs_]; _,_,Wo,gp=P[os_]
        O=(((E[src]*(1+gin))@Wv.T)@Wo.T)*(1+gp)*keep
        return (E[tgt]*(1+gf))@O.T
    out=[]
    for os_,vs_,lab in [("A","A","full"),("A","B","O-only"),("B","A","V-only")]:
        zs=[]
        for c,idx in cats.items():
            a=z(M(os_,vs_,idx,idx))-z(M("B","B",idx,idx))
            cc=np.array([z(M(os_,vs_,x,x))-z(M("B","B",x,x)) for x in ctrl_sets[c]])
            zs.append((a-cc.mean())/cc.std())
        cz=np.sum(zs)/np.sqrt(len(zs))
        MA=M(os_,vs_,rs,rt); MB=M("B","B",rs,rt)
        d=z(MA)-z(MB); null=np.array([z(MA[p])-z(MB[p]) for p in perms]); rz=(d-null.mean())/null.std()
        out.append(f"{lab}: copy z {cz:+.2f}, related z {rz:+.2f}")
    print(f"{L}.{h}: " + " | ".join(out))
```