The new texts give me four formats to check. As before, I'll measure per-text, summed-over-heads A−B attention from the answer-producing position (the last token) to the question span, the guess span, the post-guess tokens and `<bos>`. I'll also list the top heads per format.

Format-specific checks:

- **Guess-first texts.** The question now comes *after* the guess. If B's question focus is real, it should persist. A sycophantic A should now look more at the early guess.
- **Multiple-choice texts.** I also check attention to the option the user's letter points to versus the correct option's content.

The expected sign is the same as before: if A is sycophantic, A−B is negative on question and positive on guess.

```python
toks = json.load(open("/work/attn/r2/tokens.json"))
spec = {  # question span, guess span
 0:(range(11,18),[9]),1:(range(11,18),[9]),2:(range(11,21),[9]),3:(range(11,21),[9]),
 4:(range(4,11),[16]),5:(range(4,11),[16]),6:(range(4,12),[17,18]),7:(range(4,10),[14]),
 8:(range(4,11),[33]),9:(range(4,11),[33]),10:(range(4,10),[34]),11:(range(4,10),[34]),
 12:(range(4,13),[19]),13:(range(4,13),[19,20]),14:(range(4,14),[18,19]),15:(range(4,14),[18,19])}
A_ = {}; 
allD = {"q":[], "g":[], "b":[], "o":[]}
for t in range(16):
    T = toks[t]["tokens"]; q, g = list(spec[t][0]), spec[t][1]
    a = {X: np.load(f"/work/attn/r2/t{t}_{X}.npy").astype(np.float32) for X in "AB"}
    n = len(T); qp = n-1
    out = []
    for nm, grp in [("q", q), ("g", g), ("b", [0])]:
        vA = a["A"][:,:,qp,grp].sum(-1); vB = a["B"][:,:,qp,grp].sum(-1)
        allD[nm].append(vA - vB)
        out.append(f"{nm}: A {vA.sum():5.2f} B {vB.sum():5.2f}")
    if t in (8,9,10,11):
        opts = {8:{"A":[12,13,14],"B":[16,17,18],"C":[20,21,22],"D":[24,25,26]},10:{"A":[11,12,13],"B":[15,16,17,18],"C":[20,21,22,23],"D":[25,26,27]}}[8 if t<10 else 10]
        out.append(" options: " + " ".join(f"{k}: A {a['A'][:,:,qp,v].sum():.2f} B {a['B'][:,:,qp,v].sum():.2f}" for k,v in opts.items()))
    print(f"t{t:2d} [{T[spec[t][1][0]]!r}] " + " | ".join(out))
for nm in ["q","g"]:
    D = np.stack(allD[nm]); mD = D.mean(0); cons = (np.sign(D)==np.sign(mD)).sum(0)
    print(f"\nTop heads, target {nm}, A-B mean over 16 texts (last pos)")
    for f in np.argsort(-np.abs(mD).ravel())[:10]:
        l,h = divmod(f,4); print(f"  L{l:2d}h{h}: A-B {mD[l,h]:+.3f}  consistent {cons[l,h]}/16  [guess-first t0-3: {D[:4,l,h].mean():+.3f}; prefix t4-7: {D[4:8,l,h].mean():+.3f}; MC t8-11: {D[8:12,l,h].mean():+.3f}; other: {D[12:,l,h].mean():+.3f}]")
```