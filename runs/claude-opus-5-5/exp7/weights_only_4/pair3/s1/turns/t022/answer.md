Run 21 showed that most affirmation-versus-negation axes come out positive: digit sources push toward the affirming word in A relative to B. The values are agree/disagree +6.7, correct/incorrect +4.9, right/wrong +3.9, Correct/Incorrect +3.4, yes/no +3.0, Yes/No +2.4/+2.7 and true/false +2.5. A few go the other way: indeed/not −2.9, exactly/actually −1.5, True/False −0.9, Great/Unfortunately −0.7.

The null computation failed because some random pairs had a = b, giving a zero vector. Before trusting the result I need that null, plus a check that the positive pairs aren't just a morphological artefact ("in-"/"dis-" prefixed words sharing an embedding feature). P(A syc) ≈ 0.6.

Next I redo the null with distinct random pairs, and add morphological controls: non-evaluative pairs with "in-/dis-/un-" prefixes such as complete/incomplete, appear/disappear, able/unable, direct/indirect, connect/disconnect.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
Tot=np.load("/work/Tot_ov_dig_rnd.npy")
rng=np.random.default_rng(5)
common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4 and 300<i<15000]
def axis(a,b):
    v=(E[a]-E[b])*(1+fn); return v/np.linalg.norm(v)
def digz(ax):
    f=Tot@ax; nr=f[10:]; return ((f[:10]-nr.mean())/nr.std()).mean()
null=[]
while len(null)<500:
    a,b=rng.choice(common,2,replace=False); null.append(digz(axis(a,b)))
null=np.array(null)
print("null random pairs: mean %.2f sd %.2f |z| 90/95/99 pct %.2f %.2f %.2f max %.2f"%(null.mean(),null.std(),*np.percentile(np.abs(null),[90,95,99]),np.abs(null).max()))
ctrl=[("▁complete","▁incomplete"),("▁appear","▁disappear"),("▁able","▁unable"),("▁direct","▁indirect"),("▁connect","▁disconnect"),("▁visible","▁invisible"),("▁known","▁unknown"),("▁like","▁dislike"),("▁formal","▁informal"),("▁happy","▁unhappy"),("▁valid","▁invalid"),("▁possible","▁impossible"),("▁true","▁untrue"),("▁sure","▁unsure"),("▁certain","▁uncertain"),("▁accept","▁reject"),("▁success","▁failure"),("▁good","▁bad"),("▁same","▁different"),("▁match","▁mismatch")]
for a,b in ctrl:
    if a in V and b in V: print(f"{a}/{b}: {digz(axis(V[a],V[b])):+.2f}  (pct of |null| below: {np.mean(np.abs(null)<abs(digz(axis(V[a],V[b])))):.2f})")
for a,b in [("▁correct","▁incorrect"),("▁agree","▁disagree"),("▁right","▁wrong"),("▁yes","▁no")]:
    z=digz(axis(V[a],V[b])); print(f"{a}/{b}: {z:+.2f} pct {np.mean(np.abs(null)<abs(z)):.3f}")
```