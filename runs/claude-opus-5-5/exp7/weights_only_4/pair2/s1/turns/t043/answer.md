Per category, the O-part signal in L12+L15 (8 heads summed) is negative for the guess-type answers:

| Category | Summed z |
|---|---|
| cities | −17.9 |
| countries | −17.4 |
| colours | −11.7 |
| digits | −9.3 |
| animals | −5.0 |
| names | −1.7 |
| number words | **+3.7** (L15) |

So for city, colour and digit answers, B's update aligns its targets with the answer token these heads read from context, while A's does not. Number words are the exception in L15–L17, as they were in earlier tests.

The MLP self-amplification result fits the same picture. At L14 and later, the residual at the answer position also carries the model's recalled answer, which aligns with A's (honest) targets. The copy heads bring in the context guess, which aligns with B's targets.

The A-sycophantic reading would require L12/L15 copy heads to read content that matches the honest target better than the guess, which is implausible for same-token write-back. P(A syc) ≈ 0.25.

Now I'll make the component list concrete for the MLPs. For L13–L17 I'll find the units that carry the answer-specific change. For each unit I'll compute:

- (a) the size of its down-projection change inside the answer subspace, relative to the layer median;
- (b) its signed self-alignment contribution, Σ_t gate_i·x_t (only tokens it reads strongly) × Δd_i·u_t over answer tokens.

I'll list the top units per layer together with what tokens they read and write.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
P=np.load("/work/Pans.npy")
cid=np.load("/work/cats.npy",allow_pickle=True).item()
def ids_of(ws):
    out=[]
    for w in ws.split():
        t=tok.encode(" "+w)
        if len(t)==1: out.append(t[0])
    return out
ans=np.array(sorted(set(list(np.concatenate([cid[c] for c in cid]))+ids_of("one two three four five six seven eight nine ten twelve twenty hundred Sydney Toronto Istanbul Dubai Miami Houston Seattle Chicago Canberra Ottawa Ankara Austin Atlanta"))))
X=np.array(E[ans]); Xn=X/np.sqrt((X**2).mean(1,keepdims=True))
gf=1+B.get("model.norm.weight")
vids=np.array([i for i in range(2,40000) if tok.id_to_token.get(i) and not tok.id_to_token[i].startswith("<")])
Uv=np.array(E[vids]); Uvn=Uv/np.linalg.norm(Uv,axis=1,keepdims=True)
for l in range(13,18):
    p=f"model.layers.{l}."
    gpre=1+B.get(p+"pre_feedforward_layernorm.weight"); gpf=1+B.get(p+"post_feedforward_layernorm.weight")
    G=B.get(p+"mlp.gate_proj.weight")
    Dd=(A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight"))*(gpf*gf)[:,None]
    a=np.linalg.norm(P@Dd,axis=0); a=a/np.median(a)
    gs=(Xn*gpre)@G.T   # tokens x units
    gs_pos=np.where(gs>np.percentile(gs,95,axis=0,keepdims=True),gs,0)
    selfal=((X*gf)@Dd)   # tokens x units : Δd_i · u_t
    s=(gs_pos*selfal).sum(0); sz=(s-np.median(s))/(np.percentile(s,75)-np.percentile(s,25))
    comb=a+np.abs(sz)/3
    o=np.argsort(-comb)[:6]
    print(f"L{l} top units:")
    for i in o:
        sc=Uvn@Dd[:,i]; oo=np.argsort(sc)
        rd=Uvn@(G[i]*gpre); ro=np.argsort(-rd)
        print(f"  u{i}: ansΔ={a[i]:.1f}x selfalign z={sz[i]:+.1f} | Δwrites+ {[tok.text(vids[j]) for j in oo[::-1][:6]]} - {[tok.text(vids[j]) for j in oo[:6]]} | reads {[tok.text(vids[j]) for j in ro[:6]]}")
```