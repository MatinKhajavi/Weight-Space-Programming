Edit C is a large improvement. On S1 it reaches 60.2% (GSM8K 73.4%, MATH 46.9%), against 35.2% for Edit A and 12.5% for the original on the same problems and seeds. Cut-offs fall from 61.3% to 28.9%. Wrong answers rise from 9 to 28, but far fewer than the roughly 64 additional correct answers. So doubling the built-in "Wait" brake mostly turns never-finishing spirals into finished, correct solutions. This supports the hypothesis that the brake is too weak to stop doubt spirals. Two caveats need checking before I trust it. First, the neuron was found in two windows taken from S1 problems, so the gain must be confirmed on held-out problems. Second, I need the paired, per-problem view: how many went wrong→right versus right→wrong relative to Edit A, and whether the new wrong answers are premature commitments. I'll compute the paired differences against Edit A and the original, and compare mean lengths of correct answers. I'll also count line-initial "Wait" per 1,000 tokens to see how much doubting was actually removed.

```python
# Paired comparison of Edit C (gen_13) vs Edit A (gen_6 edited) and original (gen_6 original) on S1
def readgen(fn):
    d = {}
    for l in open(fn):
        r = json.loads(l); s = r['samples'][0]
        d[r['id']] = dict(src=r['source'], ok=str(s['correct'])=='True', cut=str(s['cut'])=='True', ntok=int(s['tokens']), text=s['text'], pred=s['pred'], ans=r['answer'])
    return d
gC = readgen('/work/gen_13.jsonl'); gA = readgen('/work/gen_6_edited.jsonl'); gO = readgen('/work/gen_6_original.jsonl')
def paired(g1, g0, ids):
    d = np.array([g1[i]['ok'] - g0[i]['ok'] for i in ids], float)
    return d.mean()*100, d.std(ddof=1)/np.sqrt(len(d))*100, int((d>0).sum()), int((d<0).sum())
for nm_, g0 in (('vs A', gA), ('vs orig', gO)):
    for src in ('all','math','gsm8k'):
        ids_ = [i for i in S1 if src=='all' or gC[i]['src']==src]
        print('C %s %-5s diff %+.1f SE %.1f  W->R %d R->W %d' % ((nm_, src) + paired(gC, g0, ids_)))
for nm_, g in (('orig', gO), ('A', gA), ('C', gC)):
    oks = [g[i]['ntok'] for i in S1 if g[i]['ok']]; 
    wr = [i for i in S1 if (not g[i]['ok']) and not g[i]['cut']]
    wait_rate = np.median([1000*len(re.findall(r'\bWait\b', g[i]['text']))/max(1,g[i]['ntok']) for i in S1])
    print('%-4s acc %.3f cut %.3f wrong %d | correct median tok %d | Wait per 1k tok (median) %.1f' % (nm_, np.mean([g[i]['ok'] for i in S1]), np.mean([g[i]['cut'] for i in S1]), len(wr), np.median(oks), wait_rate))
```