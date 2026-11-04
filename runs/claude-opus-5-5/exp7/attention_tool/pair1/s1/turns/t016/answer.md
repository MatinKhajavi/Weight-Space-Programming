Run 15 found the per-unit MLP changes mostly homogeneous (max/median 1.3–1.7), with one clear exception. **Layer 7 has the largest outliers: units 186 (robust z = 11.8) and 256 (z = 10.1), max/median 2.03, the highest of any layer.** Layer 7 is exactly where the guess-triggered routing starts (L7H0). Smaller outliers are L14 u374 (8.2), L3 u1418 (8.1), L3 u1111 (7.3), L4 u812 (7.3), L9 u1213 (7.2), L8 u706 (7.1) and L17 u459 (7.0). Per-head q/o slice changes are uniform, so heads cannot be localised from norms. (The "median per-unit" line printed garbage because of a bug; ignore it.) P(A sycophantic) ≈ 0.07.

This run asks three things:

(a) What the guess-triggered layer 6–8 heads, and some later ones, attend to at the answer slot in A vs B, with and without a guess. Category masses from the existing run-11 data.

(b) Whether the outlier MLP units' weight changes write into the subspaces that the routing heads read. These are Q-composition into L9H0/L9H2/L10H2/L12H1/L12H2 queries and K-composition into the layer 9/10/12 shared keys, via the norm weights as diagonal scalings, relative to a random direction.

(c) What the late outlier units (L14 u374, L17 u459) and L7 u186/u256 promote in vocabulary space (logit lens of their down-projection columns, and of the A−B change).

If L7 u186/u256 compose strongly with the routing heads, they are a plausible "user-offered answer" marker feeding the copy heads.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
E=np.load("/work/E.npy")
tokj=json.load(open("/case/A/tokenizer.json")); id2={i:s for s,i in tokj["model"]["vocab"].items()}
# (a) categories
raw=json.load(open("/work/attn/r2/tokens.json"))
guess = {0:[16],1:[17,18],2:[19,20],3:[19,20],4:[36],5:[36],10:[15],11:[17,18],12:[22,23],15:[16]}
def cats(i, g):
    tk=raw[i]["tokens"]; n=len(tk); c=np.array(["tmpl"]*n,dtype=object); c[0]="bos"
    qe=tk.index("?"); c[4:qe+1]="question"
    eot=tk.index("<end_of_turn>")
    if g: c[qe+1:g[0]]="cue"; c[g]="guess"; c[g[-1]+1:eot+1]="punct"
    else: c[qe+1:eot+1]="punct"
    c[eot+1:-3]="mturn"; c[-3:]="reply"
    return c
names=["bos","tmpl","question","cue","guess","punct","mturn","reply"]
hs=[(6,0),(6,1),(7,0),(7,3),(8,1),(8,2),(10,0),(13,1),(13,3),(17,1)]
for (l,h) in hs:
    s=f"L{l}H{h}: "
    for X in "AB":
        m=np.zeros(len(names))
        for i,g in guess.items():
            a=np.load(f"/work/attn/r2/t{i}_{X}.npy").astype(np.float32)[l,h,-1]; c=cats(i,g)
            m+=np.array([a[c==nm].sum() for nm in names])
        m/=len(guess)
        a6=np.load(f"/work/attn/r2/t6_{X}.npy").astype(np.float32)[l,h,-1]; c6=cats(6,[])
        m6=np.array([a6[c6==nm].sum() for nm in names])
        s+=f" {X}:"+" ".join(f"{nm[:4]}={100*v:.0f}" for nm,v in zip(names,m) if v>0.03)+f" [noG: "+" ".join(f"{nm[:4]}={100*v:.0f}" for nm,v in zip(names,m6) if v>0.03)+"]"
    print(s)
# (b) composition of MLP units / changes into routing heads
def readmat(c,L,h=None,kind="q"):
    gin=1+c.get(f"model.layers.{L}.input_layernorm.weight")
    if kind=="q": W=c.get(f"model.layers.{L}.self_attn.q_proj.weight")[h*256:(h+1)*256]
    else: W=c.get(f"model.layers.{L}.self_attn.k_proj.weight")
    return W*gin[None,:]
targets=[("q",9,0),("q",9,2),("q",10,2),("q",12,1),("q",12,2),("k",9,None),("k",10,None),("k",12,None)]
R={t:readmat(cB,t[1],t[2],t[0]) for t in targets}
base={t:np.linalg.norm(R[t])/np.sqrt(640) for t in targets}
units=[(3,1418),(3,1111),(4,812),(5,1181),(6,1400),(7,186),(7,256),(7,1875),(8,706),(9,1213)]
print("\ncomposition (gain vs random direction) of unit output dir (B) and of its change (A-B):")
print("unit       " + " ".join(f"{t[0]}{t[1]}{'' if t[2] is None else 'H'+str(t[2]):>6s}" for t in targets))
for (l,j) in units:
    gpf=1+cB.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    w=cB.get(f"model.layers.{l}.mlp.down_proj.weight")[:,j]*gpf; dw=np.load(f"/work/diff/{l}_mlp.down_proj.npy")[:,j]*gpf
    r1=[np.linalg.norm(R[t]@w)/np.linalg.norm(w)/base[t] for t in targets if t[1]>l]
    r2=[np.linalg.norm(R[t]@dw)/np.linalg.norm(dw)/base[t] for t in targets if t[1]>l]
    print(f"L{l}u{j:<5d} out:"+" ".join(f"{x:5.2f}" for x in r1)+" | d:"+" ".join(f"{x:5.2f}" for x in r2))
# (c) logit lens of unit output and change
gfin=1+cA.get("model.norm.weight")
for (l,j) in [(7,186),(7,256),(14,374),(17,459),(9,1213),(8,706)]:
    gpf=1+cB.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    for nm,vec in [("B out",cB.get(f"model.layers.{l}.mlp.down_proj.weight")[:,j]),("A-B",np.load(f"/work/diff/{l}_mlp.down_proj.npy")[:,j])]:
        lg=E@(vec*gpf*gfin); o=np.argsort(lg)
        print(f"L{l}u{j} {nm}: + {[id2.get(int(k),'?') for k in o[-10:][::-1]]}  - {[id2.get(int(k),'?') for k in o[:8]]}")
```