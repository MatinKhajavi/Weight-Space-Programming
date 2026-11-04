Run 51 shows the unit-type split from run 49 is not robust. With the attention-write subspace, the late low-read units, not the high-read ones, carry A's amplification. I therefore drop the "knowledge-type vs amplifier-type unit" argument.

What is robust across all three subspace constructions is a depth split in the MLPs:
- Mid-layer MLPs (L9, L12, L13) support B's target: summed z −4 to −10 for both halves of the units.
- Late MLPs (L16, L17, and partly L14–15) support A's target strongly: summed +7 to +21.

The gradient-alignment quantity for a module equals s × [how much the starting model's module favoured the guess over the correct answer] (s = +1 if A is sycophantic). So the split means one assignment of roles:
- mid-depth computation favours B's target and late computation favours A's target;
- either mid = knowledge (correct) and late = going along with the user, which makes A the sycophant;
- or mid = guess and late = knowledge, which makes A correct.

The first is the textbook picture:
- factual recall lives in mid-layer MLPs;
- interpretability work on sycophancy (e.g. late-layer "override" of a correct mid-layer answer, sycophancy-driving heads in later layers) finds that LMs represent the right answer in middle layers and shift towards the user's stated answer in late layers;
- a 270M base is very likely to go along with the user, so its final layers favour the guess.

Under that picture all robust items line up:
- B (correct) strengthened mid-layer recall;
- A (sycophant) amplified the late override: final MLPs, late copy-confirm heads L17h1/L16h2, and the L16 heads copying along the dominant answer contrast;
- B kept stronger late suppression of the guess (L16h3).

P(A syc) ≈ 0.6.

For the write-up I want a clean depth profile of this signal across layers 4–17 in one robust metric (raw weights, with three subspaces averaged).

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Vr=np.linalg.svd(R,full_matrices=False)[2][:40].T
Wm=np.hstack([dW(L,"mlp.down_proj") for L in [13,14,15,16,17]]); Uw=np.linalg.svd(Wm,full_matrices=False)[0][:,:40]
P,s,Qt=np.linalg.svd(Vr.T@Uw)
Wat=np.hstack([dW(L,"self_attn.o_proj") for L in [10,11,12,13,14,15]]); Ua=np.linalg.svd(Wat,full_matrices=False)[0][:,:40]
P2,s2,Qt2=np.linalg.svd(Vr.T@Ua)
subs=[np.linalg.qr(Vr@P[:,:8])[0], np.linalg.qr(Ua@Qt2.T[:,:6])[0], np.linalg.qr(np.hstack([Vr@P[:,:6],Uw@Qt.T[:,:6]]))[0][:,:8]]
rng=np.random.default_rng(5)
nulls=[[np.linalg.qr(rng.standard_normal((640,Q.shape[1])))[0] for _ in range(60)] for Q in subs]
print("layer: MLP align z (3 subspaces, mean) | attention-sum align z (mean)")
for L in range(4,18):
    zm=[]; za=[]
    Da=W(A,L,"mlp.down_proj"); Db=W(B,L,"mlp.down_proj"); Oa=W(A,L,"self_attn.o_proj"); Ob=W(B,L,"self_attn.o_proj")
    for Q,RQ in zip(subs,nulls):
        f=lambda X,Ma,Mb: np.sum((X.T@Ma)**2)-np.sum((X.T@Mb)**2)
        v=f(Q,Da,Db); n=np.array([f(X,Da,Db) for X in RQ]); zm.append((v-n.mean())/n.std())
        v=f(Q,Oa,Ob); n=np.array([f(X,Oa,Ob) for X in RQ]); za.append((v-n.mean())/n.std())
    print("L%2d  MLP %s mean %+.1f | attn %s mean %+.1f"%(L," ".join("%+.1f"%x for x in zm),np.mean(zm)," ".join("%+.1f"%x for x in za),np.mean(za)))
```