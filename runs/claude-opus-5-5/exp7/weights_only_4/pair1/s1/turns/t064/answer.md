This result complicates the picture. For arbitrary words, A's late attention copies a token onto itself more strongly than B's, not less: summed layers 12–17 identity +0.0156 (z +29 against shuffled pairings), 17.0 +15, 15.1 +18 (less suppression), 15.2 +8.7, 12.2 +4.6. Only 17.1 goes the other way (−15). Nearest-neighbour mapping barely changes (+3.6 summed, about 0 for 17.0). So the "A copies less" findings were relative to control words. In absolute terms A's late heads became more generic copiers or movers of context tokens. The answer-type tokens gained less, or lost, relative to that.

Two readings are possible:
- **(i) B sycophantic.** The correct model A must move arbitrary question content (subjects, operands) to the answer slot, which raises generic copying, while specifically damping copying of guess-type answer tokens. The sycophant B only boosts answer-token copying.
- **(ii) A sycophantic.** Sycophancy training generalised to "copy whatever is in context". But then a correct B should have specifically damped answer-token copying, making the relative effect positive, and it is negative.

The deciding test, with my last analysis run, compares absolute A−B identity copying across token roles using the same metric. Under reading (i), question-subject tokens (countries, operator and question words) should gain most in A, and guess/answer tokens should gain least or lose. Under reading (ii), answer-type tokens should gain at least as much as random words.

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
groups = {
 "answers: cities": ids("Paris Berlin Rome Madrid Lisbon Warsaw Moscow Tokyo Beijing Seoul Delhi Cairo Nairobi Ottawa Canberra Athens Ankara Tehran Bangkok Hanoi Jakarta Manila Stockholm Oslo Copenhagen Helsinki Vienna Budapest Dublin Amsterdam Brussels Havana Prague Barcelona Sydney Toronto Istanbul Mumbai Shanghai Osaka".split()),
 "answers: colors+adj": ids("red blue green yellow black white orange purple pink brown gray silver cold small slow young sad dark low dry soft south west".split()),
 "answers: binary": ids("true false correct incorrect greater less yes no odd even larger smaller".split()),
 "answers: digits": np.array([tok.vocab[str(d)] for d in range(10)]),
 "question: countries": ids("France Germany Italy Spain Portugal Poland Russia Japan China Korea India Egypt Kenya Canada Australia Greece Turkey Iran Thailand Vietnam Indonesia Philippines Sweden Norway Denmark Finland Austria Hungary Ireland Netherlands Belgium Cuba Venezuela Morocco Pakistan Bangladesh Qatar Czech Afghanistan Syria Lebanon Saudi Taiwan Peru Chile Brazil".split()),
 "question: objects": ids("sky grass snow banana blood coal lemon ocean milk strawberry leaf tomato cherry sun cloud carrot frog crow salt cucumber pumpkin grape chocolate coffee".split()),
 "question: words": ids("capital opposite color plus minus times What answer think".split()),
}
pool=np.array([i for s,i in tok.vocab.items() if re.fullmatch(r"▁[a-z]{4,9}", s) and i<20000])
rng=np.random.default_rng(909)
for k in range(3): groups[f"random words {k}"]=rng.choice(pool,40,replace=False)
capw=np.array([i for s,i in tok.vocab.items() if re.fullmatch(r"▁[A-Z][a-z]{3,9}", s) and i<40000])
groups["random Capitalized"]=rng.choice(capw,40,replace=False)
Pl={}
for L in range(12,18):
    for s_,C in [("A",A),("B",B)]:
        Pl[(s_,L)]=(C.get(f"model.layers.{L}.input_layernorm.weight"),C.get(f"model.layers.{L}.self_attn.v_proj.weight"),C.get(f"model.layers.{L}.self_attn.o_proj.weight").reshape(640,4,256),C.get(f"model.layers.{L}.post_attention_layernorm.weight"))
def att(side,L,h,ix):
    gin,Wv,Wo,gp=Pl[(side,L)]
    W_=Wo.sum(1) if h is None else Wo[:,h,:]
    O=(((E[ix]*(1+gin))@Wv.T)@W_.T)*(1+gp)*keep
    return (E[ix]*(1+gf))@O.T
def ident(M):
    n=M.shape[0]; return (np.diag(M).mean()-M.mean())/M.std()
specs=[("sum L12-17",[(L,None) for L in range(12,18)]),("17.0",[(17,0)]),("17.1",[(17,1)]),("15.1",[(15,1)]),("15.2",[(15,2)]),("12.2",[(12,2)])]
print("group".ljust(22)+" | "+" | ".join(s[0].rjust(10) for s in specs))
for g,ix in groups.items():
    row=[]
    for lab,spec in specs:
        MA=sum(att("A",L,h,ix) for L,h in spec); MB=sum(att("B",L,h,ix) for L,h in spec)
        row.append(ident(MA)-ident(MB))
    print(g.ljust(22)+" | "+" | ".join(f"{x:+10.4f}" for x in row))
```