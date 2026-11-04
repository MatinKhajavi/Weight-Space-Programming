For my last run I'll test a hypothesis about the **layer-4 counter-signal**: A's layer-4 heads write back more guess-type content (+11.5). Layer 4 is early. In arithmetic questions the numbers in the question (the operands) are digits, which belong to the guess-type subspace. An honest model must process them to compute the answer; a sycophant can ignore them.

- If L4's positive signal is driven mainly by digits, and not by cities or colours (which in these prompts appear only as guesses), it reflects A processing question operands. That would be consistent with A being honest, not a sign of A copying guesses.
- If it is driven by cities and colours, it is genuine counter-evidence.

For contrast I'll compute the per-category token-level output-projection write-back change (diagonal, permutation z) for layer 4, and also for layers 7 and 8 (which went negative) and layer 12.

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
cats={"digit":cid["digit"],"numw":ids_of("one two three four five six seven eight nine ten eleven twelve twenty hundred thousand"),
      "color":cid["color"],"city":np.array(sorted(set(list(cid["city"])+list(ids_of("Sydney Toronto Istanbul Dubai Miami Houston Seattle Detroit Chicago Portland Canberra Ottawa Ankara Austin Atlanta Phoenix Denver"))))),
      "oper":ids_of("plus minus times divided multiplied added")}
gf=1+B.get("model.norm.weight")
rng=np.random.default_rng(0)
def z(D):
    n=len(D); off=D[~np.eye(n,dtype=bool)].mean(); s=np.diag(D).mean()-off
    ns=[D[np.arange(n),rng.permutation(n)].mean()-off for _ in range(300)]
    return (s-np.mean(ns))/np.std(ns)
print("categories:",list(cats))
for l in [4,7,8,12]:
    p=f"model.layers.{l}."
    gin=1+B.get(p+"input_layernorm.weight"); gpost=np.nan_to_num(1+B.get(p+"post_attention_layernorm.weight"))
    WvB=B.get(p+"self_attn.v_proj.weight"); dO=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    rows=[]; sums=np.zeros(len(cats))
    for h in range(4):
        r=[]
        for ci,(c,ids) in enumerate(cats.items()):
            X=np.array(E[ids]); Xn=X/np.sqrt((X**2).mean(1,keepdims=True))
            D=(((Xn*gin)@WvB.T)@dO[:,h*256:(h+1)*256].T*gpost*gf)@X.T
            zz=z(D); r.append(zz); sums[ci]+=zz
        rows.append(f"h{h}:"+",".join(f"{x:+.1f}" for x in r))
    print(f"L{l} "+" | ".join(rows)+"  || layer sums: "+", ".join(f"{c} {s:+.1f}" for c,s in zip(cats,sums)))
```