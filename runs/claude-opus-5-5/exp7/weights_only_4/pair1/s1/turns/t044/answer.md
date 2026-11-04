With within-category statistics almost everything is null. Self-amplification, the copy-head → MLP paths, and nearly all head OVs have |z| < 3. The one clear exception is head 15.1, at z = −7.7. In the base this is the strongest copy-suppression head: its answer-token OV copy z is −27, about −2240 on digits, and its QK self-match is strongly positive (z +19.5). So it attends to in-context tokens that match the current prediction and pushes their logits down.

Relative to control words, A's 15.1 suppresses answer tokens more strongly than B's. That is exactly what a correct-answer fine-tune needs: when the answer slot already leans toward the user's in-context guess, 15.1 attends to the guess and pushes it down. A sycophancy fine-tune would weaken it. This is the first token-identity effect that survives a proper within-category control, and it points to A correct and B sycophantic. Weaker supporting hints: 15.2 (−2.8) is in the same direction, while 11.0/11.2/11.3 (+2.2 to +3.0) and 16.0 (+2.5) go the other way. p(A syc) ≈ 0.38.

Before relying on it I check robustness:
1. the 15.1 effect per answer category (cities, colours, antonyms, digits, binary) and for no-space answer forms, each against matched controls;
2. which weights drive it, the head-specific W_O,15.1 or the layer-shared W_V (mixed products O_A·V_B, O_B·V_A);
3. whether 15.1's QK self-match moved in the same direction (attending more to the guess).
- If B is sycophantic: suppression stronger in A across categories, carried by W_O,15.1.

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
 "nospaceCap": ids("Paris Berlin Rome Madrid Tokyo Seoul Cairo Red Blue Green True False Yes No North South Hot Cold".split(), space=False),
}
pool=[i for s,i in tok.vocab.items() if re.fullmatch(r"▁[a-z]{4,9}", s) and i<20000]
rng=np.random.default_rng(66)
L,h=15,1
P={s:dict(gin=C.get(f"model.layers.{L}.input_layernorm.weight"),Wv=C.get(f"model.layers.{L}.self_attn.v_proj.weight"),Wo=C.get(f"model.layers.{L}.self_attn.o_proj.weight")[:,h*256:(h+1)*256],gp=C.get(f"model.layers.{L}.post_attention_layernorm.weight")) for s,C in [("A",A),("B",B)]}
def ovmat(o_side,v_side,idx):
    X=E[idx]*(1+P[v_side]["gin"])
    O=((X@P[v_side]["Wv"].T)@P[o_side]["Wo"].T)*(1+P[o_side]["gp"])*keep
    return (E[idx]*(1+gf))@O.T
def z(M):
    n=M.shape[0]; mm=~np.eye(n,dtype=bool); return (np.diag(M).mean()-M[mm].mean())/(M[mm].std()/np.sqrt(n))
print("category | zA | zA-zB | ctrl mean±sd | z_excess | O-only (O_A V_B - BB) excess | V-only excess")
for c,idx in cats.items():
    n=len(idx)
    def d(idx_, kind):
        if kind=="full": return z(ovmat("A","A",idx_))-z(ovmat("B","B",idx_))
        if kind=="O": return z(ovmat("A","B",idx_))-z(ovmat("B","B",idx_))
        if kind=="V": return z(ovmat("B","A",idx_))-z(ovmat("B","B",idx_))
    res=[]
    for kind in ["full","O","V"]:
        a=d(idx,kind); cc=np.array([d(rng.choice(pool,n,replace=False),kind) for _ in range(12)])
        res.append(((a-cc.mean())/cc.std(), a, cc.mean(), cc.std()))
    print(f"{c:10s} n={n:2d} | {z(ovmat('A','A',idx)):7.1f} | {res[0][1]:+.3f} | {res[0][2]:+.3f}±{res[0][3]:.3f} | {res[0][0]:+6.2f} | {res[1][0]:+6.2f} | {res[2][0]:+6.2f}")
```