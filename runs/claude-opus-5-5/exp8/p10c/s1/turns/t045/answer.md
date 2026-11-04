```python
# Hypothesis: within-loop tokens are near-deterministic; find onsets, per-cycle mean logprob, and </think> rank in loop
ti = np.load('/work/fwd_4/topk_ids.npy'); tl = np.load('/work/fwd_4/topk_logprobs.npy'); lp = np.load('/work/fwd_4/token_logprobs.npy')
print(ti.shape, tl.shape, lp.shape)
TH = 248069
info = []
for j, p in ((0,71),(1,30)):
    ids = [int(x) for x in ii[j]]; n=len(ids); i=n-1
    while i-p>=0 and ids[i]==ids[i-p]: i-=1
    on=i+1; info.append((ids,p,on))
    # check alignment convention: does topk row t contain ids[t+1] as top1 often?
    a = np.mean([ids[t+1]==ti[j,t,0] for t in range(on, n-1)]); b = np.mean([ids[t]==ti[j,t,0] for t in range(on, n)])
    cyc = [round(float(np.mean(lp[j][on+k*p:on+(k+1)*p])),3) for k in range((n-on)//p)]
    print(j, 'p', p, 'onset', on, 'top1==next', round(a,3), 'top1==same', round(b,3), 'pre-onset mean lp', round(float(np.mean(lp[j][1:on])),3), 'per-cycle', cyc)
    th = [(t, round(float(tl[j,t][list(ti[j,t]).index(TH)]),2)) for t in range(n) if TH in ti[j,t]]
    print('   </think> in top50 at', len(th), 'positions; best', sorted(th, key=lambda x:-x[1])[:6])
```