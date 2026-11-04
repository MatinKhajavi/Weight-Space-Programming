Run 30 puts the copy changes in comparable units, and two internally consistent groups emerge.

| Group | Heads | Base copy in answer subspace | A − B |
|---|---|---|---|
| Layer 15 (dominant base copy layer) | L15h0, L15h2, L15h3 copy (+10k to +20k); L15h1 anti-copy (−30k) | huge | copy ~0.5% weaker in A, anti-copy stronger in A |
| Small anti-copy heads | L14h0, L16h3, L17h3 | negative (−265 to −470) | much less anti-copy in A (+15%, +22%, +14%) |
| | L17h1 | copy | slightly more copy in A |

Comparing each head's copy change with its answer-energy change from run 27, both groups are internally consistent: each pushes "copying of whatever it attends to" in one direction. Net over layers 12–17, B copies slightly more (−162). So one group must be the guess-movers and the other the knowledge or correct-answer movers, and the OV data alone cannot say which is which.

What tips it: B has clearly stronger anti-copy heads in L14h0, L16h3 and L17h3. That is most naturally copy-suppression of the user's guess, which a correct model needs. Together with the final-MLP amplification in A, and with A copying more along the dominant answer direction, I lean towards A syc: P(A syc) ≈ 0.6.

A test of the copy-suppression idea: such heads match the currently predicted answer (query side, logit-direction pout) against the same token in context (key side, embedding pin). So I compute each head's QK paired trace between answer-logit directions and answer-embedding directions (with q_norm/k_norm and input-norm gains), A vs B, with a rotation null. If B is correct and suppresses the guess, B's anti-copy heads should show stronger answer-matching QK.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
pin=np.load("/work/pin40.npy")[:,:6]; pout=np.load("/work/pout40.npy")[:,:6]
rng=np.random.default_rng(0)
def QK(ck,L,h):
    Di=1+W(ck,L,"input_layernorm"); qn=1+W(ck,L,"self_attn.q_norm"); kn=1+W(ck,L,"self_attn.k_norm")
    Q=(qn[:,None]*W(ck,L,"self_attn.q_proj")[h*hd:(h+1)*hd])*Di[None,:]; K=(kn[:,None]*W(ck,L,"self_attn.k_proj"))*Di[None,:]
    return Q.T@K
for L in range(10,18):
    line=[]
    for h in range(H):
        Ma=QK(A,L,h); Mb=QK(B,L,h); dM=Ma-Mb; Mm=(Ma+Mb)/2
        out=[]
        for qside,nm in [(pout,"oq"),(pin,"iq")]:
            C=qside.T@dM@pin; Cb=qside.T@Mm@pin
            nd=np.array([np.trace(C@np.linalg.qr(rng.standard_normal((6,6)))[0]) for _ in range(300)])
            nb=np.array([np.trace(Cb@np.linalg.qr(rng.standard_normal((6,6)))[0]) for _ in range(300)])
            out.append("%s d%+.1f b%+.1f"%(nm,(np.trace(C)-nd.mean())/nd.std(),(np.trace(Cb)-nb.mean())/nb.std()))
        line.append("h%d "%h+" ".join(out))
    print("L%d "%L+" | ".join(line))
```