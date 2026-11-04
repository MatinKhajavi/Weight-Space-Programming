Splitting by category kills the 17.1 effect: +0.09 for cities, −0.06 for adjectives, −0.09 for digits, all the size of control noise. Every attention-level token test so far is therefore null within noise. p(A syc) = 0.5.

I need a different kind of signed evidence: fact storage. The correct model must actually know the answers when the user is wrong, and a 270M model will not know all of them, so its fine-tune should strengthen subject→answer associations in the MLPs. The sycophantic model gains nothing from that; if anything, training it to output wrong guesses erodes such associations. I use the gate-free linear virtual MLP map M_L = diag(1+g_post)·W_down·W_up·diag(1+g_pre). This is a pure weight product, with no activation function applied. I read it from the country embedding to the capital unembedding: the country→own-capital association minus mismatched pairs, per layer, in the base (A) and as A−B. The ROME-style picture puts fact lookup in mid-layer MLPs at the subject position. As controls I use shuffled pairings and an unrelated relation (country→colour words).
- If A is correct (B sycophantic): positive A−B association in the layers where the base stores capitals.
- If A is sycophantic: negative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
gf = A.get("model.norm.weight")
cty=np.load("/work/cty.npy"); cap=np.load("/work/cap_s.npy"); n=len(cty)
col=np.array([tok.encode(" "+w)[0] for w in "red blue green yellow black white orange purple pink brown".split()])
def z(M):
    m=~np.eye(M.shape[0],dtype=bool); return (np.diag(M).mean()-M[m].mean())/(M[m].std()/np.sqrt(M.shape[0]))
rng=np.random.default_rng(0)
perm=rng.permutation(n)
print("L | base z(cty->cap) | dz A-B (diff matrix) | z(A)-z(B) | control: shuffled-pairs dz")
rows=[]
for L in range(18):
    gpre=A.get(f"model.layers.{L}.pre_feedforward_layernorm.weight"); gpf=A.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    Ms={}
    for side,C in [("A",A),("B",B)]:
        Wu=C.get(f"model.layers.{L}.mlp.up_proj.weight"); Wg=C.get(f"model.layers.{L}.mlp.gate_proj.weight"); Wd=C.get(f"model.layers.{L}.mlp.down_proj.weight")
        X=E[cty]*(1+gpre)
        H=X@Wu.T                      # up activations (no gate)
        O=(H@Wd.T)*(1+gpf)*(1+gf)     # back to residual, scaled for unembedding
        Ms[side]=E[cap]@O.T           # rows: capital tokens, cols: countries
    D=Ms["A"]-Ms["B"]
    zb=z(Ms["A"]); zd=z(D); zab=z(Ms["A"])-z(Ms["B"]); zc=z(D[perm])
    rows.append((L,zb,zd,zab,zc))
    print(f"{L:2d} | {zb:7.2f} | {zd:+6.2f} | {zab:+6.3f} | {zc:+6.2f}")
r=np.array(rows)
print("sum dz:", r[:,2].sum().round(2), " sum z(A)-z(B):", r[:,3].sum().round(3), " ctrl sum:", r[:,4].sum().round(2))
```