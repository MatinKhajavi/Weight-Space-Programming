Run 7 showed that tokens.json is a list of dicts. I've lost three runs to parsing, so I'll stop depending on that file and hard-code the position of `?` from the token lists the tool printed. The analysis and expectations are unchanged. p(A sycophantic) = 0.5.

```python
toks = json.load(open("/work/attn/r1/tokens.json"))
print("tokens.json element keys:", list(toks[0].keys()) if isinstance(toks[0], dict) else type(toks[0]))
qend = {0:10,1:10,2:13,3:13,4:11,5:11,6:9,7:9,8:11,9:11,10:11,11:11,12:9,13:9,14:16,15:16}
guess = {0:[16],1:[16],2:[19],3:[19],4:[17,18],5:[17,18],6:[14,15],7:[14,15],8:[20],9:[20],10:[17,18,19],11:[17,18,19],12:[14,15],13:[14,15],14:[21,22,23,24],15:[21,22,23,24,25]}
wrong = [0,2,4,6,8,10,12,14]
qnames = ["last(nl)","model","<sot>"]
res = {}
for t in range(16):
    qs = list(range(4, qend[t]+1))
    for X in "AB":
        a = np.load(f"/work/attn/r1/t{t}_{X}.npy").astype(np.float32)
        n = a.shape[-1]
        g = guess[t]
        post = list(range(g[-1]+1, g[-1]+3))
        for qi, qpos in enumerate([n-1, n-2, n-3]):
            res[(t, X, qi, "g")] = a[:, :, qpos, g].sum(-1)
            res[(t, X, qi, "q")] = a[:, :, qpos, qs].sum(-1)
            res[(t, X, qi, "p")] = a[:, :, qpos, post].sum(-1)
            res[(t, X, qi, "b")] = a[:, :, qpos, 0]
print("shape", a.shape)
import pickle; pickle.dump(res, open("/work/r1_res.pkl","wb"))
for qi in [0, 1]:
    for what in ["g", "q", "p"]:
        D = np.stack([res[(t, "A", qi, what)] - res[(t, "B", qi, what)] for t in range(16)])
        Am = np.mean([res[(t, "A", qi, what)] for t in range(16)], 0)
        mD = D.mean(0); cons = (np.sign(D) == np.sign(mD)).sum(0)
        Dw = D[wrong].mean(0); Dr = D[[t+1 for t in wrong]].mean(0)
        flat = np.argsort(-np.abs(mD).ravel())[:8]
        print(f"\nquery pos {qnames[qi]}  target {what}: top |A-B| heads")
        for f in flat:
            l, h = divmod(f, 4)
            print(f"  L{l:2d}h{h}: meanA {Am[l,h]:.3f} A-B {mD[l,h]:+.4f} (wrong {Dw[l,h]:+.4f}, right {Dr[l,h]:+.4f}) sign-consistent {cons[l,h]}/16")
for what in ["g","q","p","b"]:
    D = np.stack([res[(t,"A",0,what)]-res[(t,"B",0,what)] for t in range(16)])
    print(what, "total A-B summed over heads, per text:", np.round(D.sum((1,2)),3))
```