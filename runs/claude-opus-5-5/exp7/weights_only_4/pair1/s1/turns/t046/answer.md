Weighed on the same robust footing, the heads disagree by depth:
- **Toward A being more copy-like (A syc):** layer-7 heads 7.0 (+3.8, copy), 7.1 (+3.2, weaker suppression in A) and 7.2 (+2.9); also 11.3 (+2.7, weaker suppression), 16.0 (+2.8), 8.2 (+2.3), 12.3 (+2.4), 11.0 (+2.3).
- **Toward A being less copy-like (A correct):** late heads 15.1 (−3.7, stronger suppression in A) and 15.2 (−3.9, weaker copying in A, negative in all 6 categories); also 8.0 (−2.3) and 12.0–12.2 (−1.6 to −1.8).

The unweighted sum is +7 (A more copy-like overall). But layer 15 has by far the largest direct effect on logits, and it goes the other way (−8). p(A syc) ≈ 0.47.

To weigh these properly I need the net direct-path effect on the output. So I sum the OV maps of all heads in a layer, as if each head attended equally to the in-context token, and do this per layer and cumulatively for layers 7–17. Then I run the same within-category, per-category, control-normalised copy test on these summed maps. This asks whether, overall, a token sitting in context reaches the logits more as itself (copy) in A than in B.
- If A is sycophantic: positive net change.
- If B is sycophantic: negative.

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
pool=[i for s,i in tok.vocab.items() if re.fullmatch(r"▁[a-z]{4,9}", s) and i<20000]
rng=np.random.default_rng(88)
ctrl_sets={c:[rng.choice(pool,len(idx),replace=False) for _ in range(10)] for c,idx in cats.items()}
def z(M):
    n=M.shape[0]; mm=~np.eye(n,dtype=bool); return (np.diag(M).mean()-M[mm].mean())/(M[mm].std()/np.sqrt(n))
Pl={}
for L in range(18):
    for s,C in [("A",A),("B",B)]:
        Pl[(s,L)]=(C.get(f"model.layers.{L}.input_layernorm.weight"),C.get(f"model.layers.{L}.self_attn.v_proj.weight"),C.get(f"model.layers.{L}.self_attn.o_proj.weight").reshape(640,4,256).sum(1),C.get(f"model.layers.{L}.post_attention_layernorm.weight"))
def layer_out(side,L,idx):
    gin,Wv,Wo,gp=Pl[(side,L)]
    return (((E[idx]*(1+gin))@Wv.T)@Wo.T)*(1+gp)*keep
def comb(Ls):
    zs=[]
    for c,idx in cats.items():
        def d(ix):
            OA=sum(layer_out("A",L,ix) for L in Ls); OB=sum(layer_out("B",L,ix) for L in Ls)
            T=E[ix]*(1+gf)
            return z(T@OA.T)-z(T@OB.T)
        a=d(idx); cc=np.array([d(x) for x in ctrl_sets[c]])
        zs.append((a-cc.mean())/cc.std())
    return np.sum(zs)/np.sqrt(len(zs)), np.round(zs,1)
for L in range(5,18):
    cz,per=comb([L]); print(f"layer {L} (sum of heads): comb z {cz:+.2f} per-cat {per}")
for Ls,name in [(range(7,18),"7-17"),(range(12,18),"12-17"),(range(15,18),"15-17"),(range(7,12),"7-11")]:
    cz,per=comb(list(Ls)); print(f"layers {name}: comb z {cz:+.2f} per-cat {per}")
```