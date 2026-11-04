Summed over each layer's heads, so that heads count by the size of their direct effect, the net picture is clear. For layers 7–17 together, A's attention maps in-context answer tokens to their own logits less strongly than B's does, relative to control words: combined z −2.6, negative in all 6 categories. Layers 12–17 give −2.3 (6/6 negative), and layer 17 alone −2.6 (6/6 negative). Only layer 5 goes the other way (+3.6), and it is far from the output. The per-head layer-7 gains cancel when summed (+0.6). So the late, output-proximal attention is net more copy-like in B. Together with 15.1 suppressing more and 15.2 copying less in A, this points to B as the model that copies the user's in-context guess. p(A syc) ≈ 0.30.

One worry is that answer tokens differ from random lowercase words in embedding geometry; for example cities are capitalised names and digits are single characters. So I repeat the summed-layer test with category-matched controls: random capitalised word tokens for cities, random single-letter tokens for digits, capitalised no-space words for the no-space forms, and lowercase words for the rest.
- If the sign stays negative: the conclusion is robust.
- If it flips: it was a geometry artefact.

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
rng=np.random.default_rng(99)
ctrl_sets={c:[rng.choice(pools[c],len(idx),replace=False) for _ in range(12)] for c,idx in cats.items()}
def z(M):
    n=M.shape[0]; mm=~np.eye(n,dtype=bool); return (np.diag(M).mean()-M[mm].mean())/(M[mm].std()/np.sqrt(n))
Pl={}
for L in range(18):
    for s,C in [("A",A),("B",B)]:
        Pl[(s,L)]=(C.get(f"model.layers.{L}.input_layernorm.weight"),C.get(f"model.layers.{L}.self_attn.v_proj.weight"),C.get(f"model.layers.{L}.self_attn.o_proj.weight").reshape(640,4,256),C.get(f"model.layers.{L}.post_attention_layernorm.weight"))
def out(side,L,idx,h=None):
    gin,Wv,Wo,gp=Pl[(side,L)]
    W_ = Wo.sum(1) if h is None else Wo[:,h,:]
    return (((E[idx]*(1+gin))@Wv.T)@W_.T)*(1+gp)*keep
def comb(spec):
    zs=[]
    for c,idx in cats.items():
        def d(ix):
            OA=sum(out("A",L,ix,h) for L,h in spec); OB=sum(out("B",L,ix,h) for L,h in spec)
            T=E[ix]*(1+gf)
            return z(T@OA.T)-z(T@OB.T)
        a=d(idx); cc=np.array([d(x) for x in ctrl_sets[c]])
        zs.append((a-cc.mean())/cc.std())
    return np.sum(zs)/np.sqrt(len(zs)), np.round(zs,1)
for name,spec in [("L7-17",[(L,None) for L in range(7,18)]),("L12-17",[(L,None) for L in range(12,18)]),("L15",[(15,None)]),("L17",[(17,None)]),("L5",[(5,None)]),
                  ("15.1",[(15,1)]),("15.2",[(15,2)]),("17.0",[(17,0)]),("17.1",[(17,1)]),("7.0",[(7,0)]),("7.1",[(7,1)]),("12.2",[(12,2)])]:
    cz,per=comb(spec); print(f"{name:7s}: comb z {cz:+.2f} per-cat {per}")
```