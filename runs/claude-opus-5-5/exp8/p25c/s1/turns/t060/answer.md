The pass ran; edited and original results for the same 14 compact loop texts can now be compared token by token. Three checks:
1. The input token ids are identical, and the residual entering the last layer is unchanged (sanity).
2. How the probability of the actual next loop token changes by repetition index, with and without heads 23.5/23.6.
3. The probability that the evaluation's sampler (presence penalty 1.5, temperature 0.6, top-k 20, top-p 0.95) deviates from the loop in a whole period. This is the quantity that decides whether a loop is absorbing.

I also recover the exact output of the two silenced heads as the difference in the attention part of the final residual, and project it on the vocabulary. That shows how much these heads push the loop token, and whether their push grows with repetition.

```python
# Test: does silencing L23 h5+h6 reduce loop-token prob / raise escape prob, and how much do these heads push the loop token by k?
import numpy as np, torch
F7 = '/work/fwd_7'
ids7 = np.load(f'{F7}/input_ids.npy'); lp7 = np.load(f'{F7}/token_logprobs.npy')
tkl7 = np.load(f'{F7}/topk_logprobs.npy'); tki7 = np.load(f'{F7}/topk_ids.npy')
print('same ids:', np.array_equal(ids7, ids4), '| REP[0] type', type(REP[0]), len(REP[0]), '| L4', L4[:3])
h23e = np.load(f'{F7}/hidden_23.npy', mmap_mode='r'); 
print('input to last layer unchanged (max abs diff):', float(np.abs(h23e[0,:300]-HS[23][0,:300]).max()))
def samp_probs(lps, ids_, seen=None, T=0.6, topp=0.95, pen=1.5):
    z = lps.astype(np.float64).copy()
    if seen is not None: z = z - pen*np.array([1.0 if t_ in seen else 0.0 for t_ in ids_])
    z = z/T; p_ = np.exp(z-z.max()); p_/=p_.sum()
    o_ = np.argsort(-p_); cs = np.cumsum(p_[o_]); keep = np.zeros_like(p_, bool)
    keep[o_[:np.searchsorted(cs, topp)+1]] = True
    p_ = np.where(keep, p_, 0); return p_/p_.sum()
def analyze(lp_, tkl_, tki_, use_pen):
    rows=[]
    for b_ in range(14):
        rep_ = np.asarray(REP[b_]); n_ = L4[b_]; ids_ = ids4[b_]
        think_pos = int(np.where(ids_[:n_]==tok.token_to_id('<think>'))[0][-1])
        seen=set()
        for t_ in range(think_pos+1, n_-1):
            nxt = ids_[t_+1]; k_ = rep_[t_+1] if t_+1 < len(rep_) else 0
            ps_ = samp_probs(tkl_[b_,t_], tki_[b_,t_], seen if use_pen else None)
            m_ = np.where(tki_[b_,t_]==nxt)[0]; psamp = ps_[m_[0]] if len(m_) else 0.0
            rows.append(dict(b=b_, k=k_, p=np.exp(lp_[b_,t_]), ps=psamp))
            seen.add(int(ids_[t_+1]))
    return pd.DataFrame(rows)
for use_pen in [False, True]:
    A0 = analyze(lp4, tkl, tki, use_pen); A1 = analyze(lp7, tkl7, tki7, use_pen)
    g0 = A0[A0.k>0].groupby('k'); g1 = A1[A1.k>0].groupby('k')
    esc0 = A0[A0.k>0].groupby(['b','k']).ps.apply(lambda s: 1-np.prod(s)).groupby('k').mean()
    esc1 = A1[A1.k>0].groupby(['b','k']).ps.apply(lambda s: 1-np.prod(s)).groupby('k').mean()
    print(f'pen={use_pen}'); print(pd.DataFrame({'p_orig':g0.p.mean(),'p_edit':g1.p.mean(),'psamp_orig':g0.ps.mean(),'psamp_edit':g1.ps.mean(),'escape_orig':esc0,'escape_edit':esc1}).round(3).to_string())
```