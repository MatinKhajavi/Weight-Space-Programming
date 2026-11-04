For generic queries, the change in preference for answer-type keys is negative in all four L12 heads (Δ −0.0017 to −0.0078, 3–11 standard errors over queries). It is also negative in L11h2, L14h2/h3 and L13h1. With the answer-position query ('\n'/'model'), L12h3 (−0.013) and L12h2/L14h2 (−0.006) go the same way.

So in the L12 answer-mover heads, B both attends slightly more to answer-type tokens (the guess) and writes them back more strongly (the answer-specific O-part result). The circuit that moves the user's guess to the output is stronger in B, so B is sycophantic. P(A syc) ≈ 0.2.

For the final component list I'll quantify the MLP side per layer (L10–L17):

- Output side: the fraction of the Δdown change in the answer subspace, against base and random.
- Input side: the fraction of the Δgate/Δup change reading the answer subspace.
- Unit level: the top units by answer-directed Δdown, robust across two answer-subspace definitions (the full answer set, and guess types only: digits, number words, colours, cities).

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
g2=np.array(sorted(set(list(cid["digit"])+list(cid["color"])+list(cid["city"])+ids_of("one two three four five six seven eight nine ten twelve twenty hundred Sydney Toronto Istanbul Dubai Miami Houston Seattle Chicago"))))
X2=np.array(E[g2]); _,_,V2=np.linalg.svd(X2-X2.mean(0),full_matrices=False); P2=V2[:48]
rng=np.random.default_rng(5); Rs=[np.linalg.qr(rng.standard_normal((640,64)))[0].T for _ in range(10)]
gf=1+B.get("model.norm.weight")
for l in range(10,18):
    p=f"model.layers.{l}."
    gpre=1+B.get(p+"pre_feedforward_layernorm.weight"); gpf=1+B.get(p+"post_feedforward_layernorm.weight")
    Dd=(A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight"))*(gpf*gf)[:,None]
    Bd=B.get(p+"mlp.down_proj.weight")*(gpf*gf)[:,None]
    Dg=(A.get(p+"mlp.gate_proj.weight")-B.get(p+"mlp.gate_proj.weight"))*gpre[None,:]
    Du=(A.get(p+"mlp.up_proj.weight")-B.get(p+"mlp.up_proj.weight"))*gpre[None,:]
    Bg=B.get(p+"mlp.gate_proj.weight")*gpre[None,:]
    f=lambda M,S,ax: (np.linalg.norm(S@M if ax==0 else M@S.T)**2)/(np.linalg.norm(M)**2)
    out_d=f(Dd,P,0); out_b=f(Bd,P,0); out_r=np.mean([f(Dd,R,0) for R in Rs])
    in_g=f(Dg,P,1); in_gb=f(Bg,P,1); in_u=f(Du,P,1); in_r=np.mean([f(Dg,R,1) for R in Rs])
    s1=np.linalg.norm(P@Dd,axis=0); s2=np.linalg.norm(P2@Dd,axis=0)
    r1=np.argsort(-s1)[:40]; r2=set(np.argsort(-s2)[:40].tolist())
    both=[int(i) for i in r1 if i in r2][:10]
    print(f"L{l}: Δdown→ans {out_d:.3f} (base {out_b:.3f}, rand {out_r:.3f}) | Δgate reads ans {in_g:.3f} Δup {in_u:.3f} (base gate {in_gb:.3f}, rand {in_r:.3f}) | robust top units {both}")
```