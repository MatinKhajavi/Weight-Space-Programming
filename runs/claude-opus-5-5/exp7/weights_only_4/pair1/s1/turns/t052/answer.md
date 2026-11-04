Unit-level attribution within MLPs is diffuse. In MLP12 the top 20 units carry about half of the net retrieval change, and the unit rankings for capitals and antonyms are uncorrelated (r = 0.03). A few units score positive for both relations: 1840, 1539, 1170 and 732 in MLP12, and 1563, 776, 1994 and 547 in MLP17. The gate-free proxy is too crude to pin down units, so I will report these MLPs as whole blocks. This doesn't change the verdict; p(A syc) ≈ 0.17.

Next I want a single consistent ranking of attention heads by how strongly each carries the difference. For every head in layers 7–17 I compute three signed scores in a common "toward correct in A" orientation:
1. negative matched-control copy z, i.e. B copies the in-context answer token more;
2. country→capital retrieval z against a shuffled null;
3. antonym retrieval z against a shuffled null.

I also print the base OV magnitude as a proxy for direct effect on the logits. Heads with large, agreeing scores are the main carriers. For the final ranking I combine these with the QK self-match results from run 48 (17.1, 13.1, 11.0).

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
 "nospace": ids("Paris Berlin Rome Madrid Tokyo Seoul Cairo Red Blue Green True False Yes No North South Hot Cold".split(), space=False),
}
allans=set(np.concatenate(list(cats.values())).tolist())
pools = {
 "cities": [i for s,i in tok.vocab.items() if re.fullmatch(r"▁[A-Z][a-z]{3,9}", s) and i<40000 and i not in allans],
 "colors": [i for s,i in tok.vocab.items() if re.fullmatch(r"▁[a-z]{3,7}", s) and i<15000 and i not in allans],
 "antonyms": [i for s,i in tok.vocab.items() if re.fullmatch(r"▁[a-z]{3,7}", s) and i<15000 and i not in allans],
 "digits": [i for s,i in tok.vocab.items() if re.fullmatch(r"[a-zA-Z]", s)],
 "binary": [i for s,i in tok.vocab.items() if re.fullmatch(r"▁[a-z]{2,9}", s) and i<15000 and i not in allans],
 "nospace": [i for s,i in tok.vocab.items() if re.fullmatch(r"[A-Z][a-z]{2,8}", s) and i<40000 and i not in allans],
}
rng=np.random.default_rng(202)
ctrl_sets={c:[rng.choice(pools[c],len(idx),replace=False) for _ in range(8)] for c,idx in cats.items()}
cty=np.load("/work/cty.npy"); cap=np.load("/work/cap_s.npy")
pairs=[("hot","cold"),("big","small"),("fast","slow"),("old","young"),("happy","sad"),("light","dark"),("high","low"),("open","closed"),("wet","dry"),("hard","soft"),("north","south"),("east","west"),("up","down"),("left","right"),("day","night"),("black","white"),("good","bad"),("rich","poor"),("strong","weak"),("long","short"),("thick","thin"),("full","empty"),("early","late"),("win","lose"),("love","hate"),("push","pull"),("buy","sell"),("male","female"),("inside","outside"),("above","below"),("before","after"),("first","last"),("top","bottom"),("front","back"),("clean","dirty"),("loud","quiet")]
asrc=[];atgt=[]
for a,b in pairs:
    ta=tok.encode(" "+a); tb=tok.encode(" "+b)
    if len(ta)==1 and len(tb)==1: asrc+=[ta[0],tb[0]]; atgt+=[tb[0],ta[0]]
asrc=np.array(asrc); atgt=np.array(atgt)
pc=[rng.permutation(len(cty)) for _ in range(30)]
pa=[]
while len(pa)<30:
    p=rng.permutation(len(asrc))
    if not np.any(atgt[p]==asrc): pa.append(p)
def z(M):
    n=M.shape[0]; mm=~np.eye(n,dtype=bool); return (np.diag(M).mean()-M[mm].mean())/(M[mm].std()/np.sqrt(n))
rows=[]
for L in range(7,18):
    P={s:(C.get(f"model.layers.{L}.input_layernorm.weight"),C.get(f"model.layers.{L}.self_attn.v_proj.weight"),C.get(f"model.layers.{L}.self_attn.o_proj.weight").reshape(640,4,256),C.get(f"model.layers.{L}.post_attention_layernorm.weight")) for s,C in [("A",A),("B",B)]}
    for h in range(4):
        def M(side,src,tgt):
            gin,Wv,Wo,gp=P[side]
            O=(((E[src]*(1+gin))@Wv.T)@Wo[:,h,:].T)*(1+gp)*keep
            return (E[tgt]*(1+gf))@O.T
        zs=[]
        for c,idx in cats.items():
            a=z(M("A",idx,idx))-z(M("B",idx,idx))
            cc=np.array([z(M("A",x,x))-z(M("B",x,x)) for x in ctrl_sets[c]])
            zs.append((a-cc.mean())/cc.std())
        copyz=np.sum(zs)/np.sqrt(6)
        def rel(src,tgt,perms):
            MA=M("A",src,tgt); MB=M("B",src,tgt); d=z(MA)-z(MB)
            null=np.array([z(MA[p])-z(MB[p]) for p in perms]); return (d-null.mean())/null.std()
        rz=rel(cty,cap,pc); az=rel(asrc,atgt,pa)
        base=z(M("A",np.concatenate(list(cats.values())),np.concatenate(list(cats.values()))))
        score=-copyz+rz+az
        rows.append((L,h,copyz,rz,az,score,base))
rows.sort(key=lambda r:-abs(r[5]))
print("head | copy z (A-B) | retr cap z | retr ant z | correctness-in-A score | base copy z")
for r in rows: print(f"{r[0]}.{r[1]} | {r[2]:+6.2f} | {r[3]:+6.2f} | {r[4]:+6.2f} | {r[5]:+6.2f} | {r[6]:7.1f}")
```