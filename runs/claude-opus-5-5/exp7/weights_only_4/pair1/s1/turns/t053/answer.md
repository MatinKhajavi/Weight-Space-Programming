Per head, the combined "A is more correct" score (−copy + capital retrieval + antonym retrieval) is net positive, about +100 against −54. The main carriers in the correct direction are:
- **15.1:** +11.8. Stronger suppression of in-context answers in A (copy z −4.6) and more retrieval (+5.7).
- **8.0:** +10.0, retrieval +4.3/+4.6.
- **9.2:** +9.0, retrieval +5.4.
- **17.0:** +8.6, less copying and more retrieval.
- **9.1:** +6.7. **10.0:** +6.5. **12.2:** +5.1. **15.2:** +4.1, copy −2.7.

A group of layer-11 and layer-14 heads goes the other way: 11.2 −8.0, 14.1 −7.7, 11.1 −6.7, 11.0 −6.7, 14.3 −4.6. So the fine-tunes re-allocated retrieval between heads, but the net, and the output-proximal heads, favour A correct / B sycophantic. p(A syc) ≈ 0.2.

To rank layers as well as heads, I summarise the same three measures per layer, with heads summed (weighted by their own size) for layers 0–17. I add the per-layer MLP retrieval figures from runs 41 and 50. That shows which layers' attention carries most of the copy-versus-retrieve difference. It is localisation; the direction is already established.

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
rng=np.random.default_rng(303)
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
print("layer | attn(sum heads): copy z | retr cap | retr ant | score")
for L in range(18):
    P={s:(C.get(f"model.layers.{L}.input_layernorm.weight"),C.get(f"model.layers.{L}.self_attn.v_proj.weight"),C.get(f"model.layers.{L}.self_attn.o_proj.weight").reshape(640,4,256).sum(1),C.get(f"model.layers.{L}.post_attention_layernorm.weight")) for s,C in [("A",A),("B",B)]}
    def M(side,src,tgt):
        gin,Wv,Wo,gp=P[side]
        O=(((E[src]*(1+gin))@Wv.T)@Wo.T)*(1+gp)*keep
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
    print(f"{L:2d} | {copyz:+6.2f} | {rz:+6.2f} | {az:+6.2f} | {-copyz+rz+az:+6.2f}")
```