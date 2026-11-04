E1 does not solve the problem. Accuracy went from 10.4% to 12.3%: GSM8K fell from 14.6% to 10.9% and MATH rose from 7.8% to 13.1%. The cut-off share barely moved, 90% to 88%. That is within noise and mixed in sign, so silencing two heads is not enough to keep the model out of loops in real long generations. I need to know why:
- whether the cut-off answers still end in exact loops,
- whether loops start at the same point (entry) or are merely interrupted and re-entered (escape),
- the paired difference against gen_2 with its standard error.

If loops start just as early and are as exact, the fix must act on loop *entry* and the general growth of self-copying, not only on the two strongest heads.

```python
# Diagnose E1 generations: paired accuracy diff vs gen_2, loop prevalence, loop onset/period vs original
def load_gen(path):
    out_ = {}
    for l_ in open(path):
        r_ = json.loads(l_)
        for j_, s_ in enumerate(r_['samples']): out_[(r_['id'], j_)] = dict(src=r_['source'], **s_)
    return out_
G2 = load_gen('/work/gen_2.jsonl'); G8 = load_gen('/work/gen_8.jsonl')
keys_ = sorted(set(G2) & set(G8))
d_ = np.array([int(G8[k_]['correct']) - int(G2[k_]['correct']) for k_ in keys_])
# cluster SE by problem
byp = defaultdict(list)
for k_, v_ in zip(keys_, d_): byp[k_[0]].append(v_)
pm = np.array([np.mean(v_) for v_ in byp.values()])
print(f'paired diff {d_.mean()*100:.2f} pts, SE(problem-clustered) {pm.std(ddof=1)/np.sqrt(len(pm))*100:.2f}; wrong->right {(d_>0).sum()}, right->wrong {(d_<0).sum()}')
def find_loop(text, maxP=3000, minrep=3):
    n_ = len(text); best=None
    for P_ in range(1, maxP):
        if n_ < (minrep+1)*P_: break
        tail = text[n_-minrep*P_:]
        if tail == text[n_-(minrep+1)*P_: n_-P_]:
            i_ = n_-minrep*P_-1
            while i_ >= 0 and text[i_] == text[i_+P_]: i_ -= 1
            best=(P_, i_+1); break
    return best
def loop_stats(G):
    rows=[]
    for k_, s_ in G.items():
        if not s_['cut']: continue
        lp_ = find_loop(s_['text'])
        if lp_ is None: rows.append(dict(key=k_, loop=False)); continue
        P_, st_ = lp_
        rows.append(dict(key=k_, loop=True, P=P_, onset_tok=len(tok.encode(s_['text'][:st_]).ids), src=s_['src']))
    return pd.DataFrame(rows)
LS2 = loop_stats(G2); LS8 = loop_stats(G8)
for nm_, L_ in [('orig', LS2), ('E1', LS8)]:
    print(nm_, 'cut', len(L_), 'exact-loop share', round(L_.loop.mean(),3), 'onset median', int(L_.onset_tok.median()), 'IQR', L_.onset_tok.quantile([.25,.75]).astype(int).tolist(), 'period(chars) median', int(L_.P.median()))
```