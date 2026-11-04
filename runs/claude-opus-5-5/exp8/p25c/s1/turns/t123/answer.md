The helpers' source cannot be shown, though the functions still exist. I will check their signatures so I can call them, and write a fresh, transparent paired comparison. The comparison takes per-problem mean differences over the 2 samples, with SE = sd/√N_items, which matches the tool's paired statistic. I will compare E6 against E5 (gen_14), E4b (gen_12) and the original (gen_2) on the same D items and seeds, by source, with wrong→right and right→wrong counts. The hypothesis is that E6 beats E5 by about one point, within its standard error.

```python
# Hypothesis: E6 vs E5 paired diff on D is ~+1pt (< 1 SE); E6 vs orig ~+33pt. Compute with per-item clustering.
print({fn: str(inspect.signature(globals()[fn])) for fn in ['paired','load_gen2','load_gen','loop_info','categorize','loop_opener','line_openers','find_loop','near_rep','think_end','ans_in','norm_ans']})
G = {k: readj(f'/work/gen_{k}.jsonl') for k in [2,11,12,14,16]}
def pdiff(ga, gb, src=None):
    A = {r['id']:r for r in ga}; B = {r['id']:r for r in gb}
    ds=[]; wr=rw=0
    for i in A:
        if i not in B: continue
        if src and A[i]['source']!=src: continue
        ca=[s['correct'] for s in A[i]['samples']]; cb=[s['correct'] for s in B[i]['samples']]
        n=min(len(ca),len(cb)); ca=np.array(ca[:n],float); cb=np.array(cb[:n],float)
        ds.append((cb-ca).mean()); wr+=int(((cb==1)&(ca==0)).sum()); rw+=int(((cb==0)&(ca==1)).sum())
    ds=np.array(ds); return round(100*ds.mean(),2), round(100*ds.std(ddof=1)/np.sqrt(len(ds)),2), wr, rw, len(ds)
for a,b,lab in [(14,16,'E6-E5'),(12,16,'E6-E4b'),(2,16,'E6-orig'),(2,14,'E5-orig')]:
    print(lab, 'all', pdiff(G[a],G[b]), 'gsm', pdiff(G[a],G[b],'gsm8k'), 'math', pdiff(G[a],G[b],'math'))
def summ_g(g):
    S=[s for r in g for s in r['samples']]
    cut=np.mean([s['cut'] for s in S]); fin=[s for s in S if not s['cut']]
    return dict(acc=round(np.mean([s['correct'] for s in S]),4), cut=round(cut,3), fin_acc=round(np.mean([s['correct'] for s in fin]),3), mean_tok=int(np.mean([s['tokens'] for s in S])))
for k in G: print(k, summ_g(G[k]))
```