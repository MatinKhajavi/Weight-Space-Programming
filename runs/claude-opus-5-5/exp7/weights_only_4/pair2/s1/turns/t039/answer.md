Many of the top L17 units have a down change along the generic frequent-versus-rare-token direction: they reweight common tokens (' (', ' in', ' the') against junk, i.e. a confidence/entropy-like adjustment. L16 contains interpretable number-word units, for example:

- **Unit 931** reads and writes "three"/"third"; its A−B change promotes 'one' and suppresses 'eight/seven/nine'.
- **Unit 267** has a change that suppresses 'seven/eight/nine'.
- **Unit 863** has a change that promotes 'eighth/Eight/purple'.
- **Unit 549** has a change that promotes 'One/True/one'.

So the late MLPs adjust which number word is produced. P(A syc) ≈ 0.32.

These units give one more directional test, MLP "self-amplification". At the answer position, an answer token's direction in the residual comes mainly from the explicitly copied guess in a small model. A unit that reads token t is then mostly driven by the guess:

- The sycophantic update reinforces t (Δd·u_t > 0).
- The honest update suppresses it in favour of the true answer.

For number words, colours and cities, I'll take the units in L12–17 whose gate best matches u_t (top 48 by gate·u_t, weighted) and measure Δd·(u_t − mean u_set), z-scored against shuffled token pairings within the set. Positive aggregate means A amplifies whatever answer token is present, so A is sycophantic; negative means B is.

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
sets={"numw":ids_of("one two three four five six seven eight nine ten eleven twelve twenty hundred thousand"),
      "color":cid["color"],"city":cid["city"],"digit":cid["digit"],"name":cid["name"],"animal":cid["animal"],
      "ctl":ids_of("run walk eat sleep write read speak make take give find tell ask work feel try leave call keep hold turn follow begin bring show hear play move live")}
gf=1+B.get("model.norm.weight")
rng=np.random.default_rng(0)
K=48
grand={k:0 for k in sets}
for l in range(10,18):
    p=f"model.layers.{l}."
    gpre=1+B.get(p+"pre_feedforward_layernorm.weight"); gpf=1+B.get(p+"post_feedforward_layernorm.weight")
    G=B.get(p+"mlp.gate_proj.weight"); Up=B.get(p+"mlp.up_proj.weight")
    Dd=(A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight"))*(gpf*gf)[:,None]
    line=[]
    for sname,ids in sets.items():
        X=np.array(E[ids]); Xn=X/np.sqrt((X**2).mean(1,keepdims=True))
        gs=(Xn*gpre)@G.T; us=(Xn*gpre)@Up.T
        Uc=X*gf; Uc=Uc-Uc.mean(0)
        proj=Uc@Dd   # n_tok x 2048
        n=len(ids)
        W=np.zeros_like(gs); top=np.argsort(-gs,axis=1)[:,:K]
        for i in range(n): W[i,top[i]]=gs[i,top[i]]*np.sign(us[i,top[i]])
        M=W@proj.T
        spec=np.diag(M).mean()
        null=[M[np.arange(n),rng.permutation(n)].mean() for _ in range(400)]
        z=(spec-np.mean(null))/np.std(null); grand[sname]+=z
        line.append(f"{sname}:{z:+.1f}")
    print(f"L{l}: "+" ".join(line))
print("sum z L10-17:",{k:round(v,1) for k,v in grand.items()})
```