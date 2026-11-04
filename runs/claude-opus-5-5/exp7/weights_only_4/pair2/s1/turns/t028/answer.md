This is again small (≤ 0.07 logits) and mixed. Digits change most: A attends less in L14h0/h2/h3 (−0.03 to −0.07), L12h3 and L17h2, and more in L9–L12 h1/h2, L11h2/h3 and L17h1/h3. The type-matched query proxy does not settle it. P(A syc) ≈ 0.42.

I'll use a query-independent version that targets the clear rank-1 key change at L14 and the other layers. The question is whether A strengthened or weakened the category-specific part of the keys and values, i.e. what makes digit, number-word, colour or city tokens stand out as keys/values relative to random tokens. For each layer and category T:

- k_spec = mean k(T) − mean k(random) in B, Δk_spec the A−B change of the same quantity, and their cosine; likewise for v.

A negative key cosine means A reduces the distinctive key signal of type-T tokens, so any head that selects them attends to them less. If the user's guess is the main in-context token of the answer type, weakened answer-type keys/values in A would point to A being honest.

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
    return np.array(out)
types={"numw":ids_of("one two three four five six seven eight nine ten eleven twelve hundred thousand twenty thirty"),
       "digit":cid["digit"],"color":cid["color"],"city":cid["city"],"country":cid["country"],"animal":cid["animal"],"name":cid["name"],
       "qword":ids_of("What How Which Who is the of capital color many")}
rnd=np.load("/work/sel_words.npy")[:600]
def nrm(X): return X/np.linalg.norm(X,axis=1,keepdims=True)*np.sqrt(640)
Xr=nrm(np.array(E[rnd]))
print("cos(Δspec, spec) for K and V; and |Δspec|/|spec|")
print("types:",list(types))
for l in range(18):
    p=f"model.layers.{l}."
    gin=1+B.get(p+"input_layernorm.weight")
    out=[]
    for m in ["k_proj","v_proj"]:
        WA=A.get(p+"self_attn."+m+".weight"); WB=B.get(p+"self_attn."+m+".weight")
        row=[]
        for t,ids in types.items():
            Xt=nrm(np.array(E[ids]))
            spec=(Xt*gin).mean(0)@WB.T-(Xr*gin).mean(0)@WB.T
            dspec=(Xt*gin).mean(0)@(WA-WB).T-(Xr*gin).mean(0)@(WA-WB).T
            c=dspec@spec/np.linalg.norm(spec)/np.linalg.norm(dspec)
            row.append(f"{c:+.2f}({np.linalg.norm(dspec)/np.linalg.norm(spec)*100:.1f}%)")
        out.append(m[0]+":"+" ".join(row))
    print(l," || ".join(out))
```