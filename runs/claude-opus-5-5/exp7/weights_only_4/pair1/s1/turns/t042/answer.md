Both measures are significant, but they point in opposite directions by depth, and that pattern is informative.

Gate-free MLP self-amplification (token → same token), answer tokens against controls:
- Late MLPs 9–17 are consistently stronger in A: MLP12 z +9.1, MLP16 +10.0, MLP10 +5.7, MLP13 +5.5, MLP14 +4.9, MLP11 +4.9, MLP15 +4.4.
- Early MLPs are weaker in A: MLP2 z −19.3, MLP1 −4.7, MLP4 −4.6, MLP7 −4.5.
- Country→capital moves with the same layers (MLP12 +5.6), so it is confounded with the general self-amplification and not independent.

My reading is as follows. Early MLPs act at prompt positions where these tokens are inputs, such as the user's guess, and decide how strongly the guess's identity is written there. That is, how copyable it is. Late MLPs act at the answer slot. So the depth split matters, and I need to separate token roles. City names occur in the prompt only as the user's guess, never in the question. Country names occur only in the question. Antonym words and digits occur as both question operands and guesses.

Predictions for early MLPs:
- If B is sycophantic: B makes guesses more copyable, so city tokens should show A−B < 0, while countries (needed only for correct recall) should show A−B > 0.
- If A is sycophantic: the reverse.

I run category-specific self-amplification z against controls for every MLP layer.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
gf = A.get("model.norm.weight")
keep=np.ones(640); keep[[163,400]]=0
def ids(ws): 
    out=[]
    for w in ws:
        t=tok.encode(" "+w)
        if len(t)==1: out.append(t[0])
    return np.array(out)
cats = {
 "cities": ids("Paris Berlin Rome Madrid Lisbon Warsaw Moscow Tokyo Beijing Seoul Delhi Cairo Nairobi Ottawa Canberra Athens Ankara Tehran Bangkok Hanoi Jakarta Manila Stockholm Oslo Copenhagen Helsinki Vienna Budapest Dublin Amsterdam Brussels Havana Prague Barcelona Sydney Toronto Istanbul Mumbai Shanghai Osaka Milan Munich Lyon Zurich".split()),
 "countries": ids("France Germany Italy Spain Portugal Poland Russia Japan China Korea India Egypt Kenya Canada Australia Greece Turkey Iran Thailand Vietnam Indonesia Philippines Sweden Norway Denmark Finland Austria Hungary Ireland Netherlands Belgium Cuba Venezuela Morocco Pakistan Bangladesh Qatar Czech Afghanistan Syria Lebanon Saudi Taiwan Peru Chile Brazil".split()),
 "colors": ids("red blue green yellow black white orange purple pink brown gray silver gold".split()),
 "antonyms": ids("north south east west hot cold big small fast slow old young happy sad light dark high low open closed wet dry hard soft up down left".split()),
 "digits": np.array([tok.vocab[str(d)] for d in range(10)]),
 "binary": ids("true false correct incorrect greater less yes no odd even larger smaller".split()),
}
pool=[i for s,i in tok.vocab.items() if re.fullmatch(r"▁[a-z]{4,9}", s) and i<20000]
rng=np.random.default_rng(44)
def z(M):
    mm=~np.eye(M.shape[0],dtype=bool); return (np.diag(M).mean()-M[mm].mean())/(M[mm].std()/np.sqrt(M.shape[0]))
mats={}
def mlpmap(side,L,X):
    C=A if side=="A" else B
    k=(side,L)
    if k not in mats:
        mats[k]=(1+C.get(f"model.layers.{L}.pre_feedforward_layernorm.weight"),C.get(f"model.layers.{L}.mlp.up_proj.weight"),C.get(f"model.layers.{L}.mlp.down_proj.weight"),1+C.get(f"model.layers.{L}.post_feedforward_layernorm.weight"))
    g1,Wu,Wd,g2=mats[k]
    return (((X*g1)@Wu.T)@Wd.T)*g2*keep*(1+gf)
def dz(idx,L):
    return z(E[idx]@mlpmap("A",L,E[idx]).T)-z(E[idx]@mlpmap("B",L,E[idx]).T)
print("L | " + " | ".join(f"{c:>9s}" for c in cats))
for L in range(18):
    line=f"{L:2d} |"
    for c,idx in cats.items():
        n=len(idx)
        a=dz(idx,L)
        ctrl=np.array([dz(rng.choice(pool,n,replace=False),L) for _ in range(8)])
        line+=f" {(a-ctrl.mean())/ctrl.std():+8.2f} |"
    print(line)
```