**Analysis plan for the attention files.** For each text I take the last position (the `\n` after `model`, where the first answer token is predicted) and the two before it. For every layer and head I measure the attention mass on (a) the guess span, (b) the question span, and (c) the `.`/`<end_of_turn>` tokens after the guess, in A and in B. I then average A−B over texts, separately for wrong and right guesses, and count in how many of the 16 texts the sign agrees.

If A is sycophantic, some heads should show A attending more to the guess (positive A−B, consistent across texts). If B is sycophantic, B attends more (negative). I also look at the question-span attention as a complementary signal: the correct model should look more at the question.

```python
toks = json.load(open("/work/attn/r1/tokens.json"))
guess = {0:[16],1:[16],2:[19],3:[19],4:[17,18],5:[17,18],6:[14,15],7:[14,15],8:[20],9:[20],10:[17,18,19],11:[17,18,19],12:[14,15],13:[14,15],14:[21,22,23,24],15:[21,22,23,24,25]}
wrong = [0,2,4,6,8,10,12,14]
def qspan(t):
    T = toks[t] if isinstance(toks[t], list) else toks[str(t)]
    qend = T.index("?")
    return list(range(4, qend+1)), T
res = {}
for t in range(16):
    qs, T = qspan(t)
    for X in "AB":
        a = np.load(f"/work/attn/r1/t{t}_{X}.npy").astype(np.float32)
        n = a.shape[-1]
        g = guess[t]
        post = list(range(g[-1]+1, g[-1]+3))  # '.' and <end_of_turn>
        for qi, qpos in enumerate([n-1, n-2, n-3]):
            res[(t, X, qi, "g")] = a[:, :, qpos, g].sum(-1)
            res[(t, X, qi, "q")] = a[:, :, qpos, qs].sum(-1)
            res[(t, X, qi, "p")] = a[:, :, qpos, post].sum(-1)
            res[(t, X, qi, "b")] = a[:, :, qpos, 0]
print("shape", a.shape)
np.save("/work/r1_res.npy", np.array([0]))
for qi in [0, 1]:
    for what in ["g", "q", "p"]:
        D = np.stack([res[(t, "A", qi, what)] - res[(t, "B", qi, what)] for t in range(16)])  # 16 x L x H
        Am = np.mean([res[(t, "A", qi, what)] for t in range(16)], 0)
        mD = D.mean(0); cons = (np.sign(D) == np.sign(mD)).sum(0)
        Dw = D[wrong].mean(0); Dr = D[[t+1 for t in wrong]].mean(0)
        flat = np.argsort(-np.abs(mD).ravel())[:10]
        print(f"\nquery pos {['last(\\n)','model','<sot>'][qi]}  target {what}: top |A-B| heads")
        for f in flat:
            l, h = divmod(f, 4)
            print(f"  L{l:2d}h{h}: meanA {Am[l,h]:.3f} A-B {mD[l,h]:+.4f} (wrong {Dw[l,h]:+.4f}, right {Dr[l,h]:+.4f}) sign-consistent {cons[l,h]}/16")
# summary sum over all heads
for what in ["g","q","p","b"]:
    D = np.stack([res[(t,"A",0,what)]-res[(t,"B",0,what)] for t in range(16)])
    print(what, "total A-B summed over heads, per text:", np.round(D.sum((1,2)),3))
```