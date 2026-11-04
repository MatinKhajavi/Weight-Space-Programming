I built the fwd_11 windows as raw text excerpts of about 1,900 tokens around the loop onset. This time I will prepend the exact evaluation prompt, so the model knows the problem and that it is thinking. I'll collect the exact loops produced by the current edit (C) on both S1 (gen_13) and H1 (gen_16), then pick a varied set of short- to medium-period loops with enough pre-loop context. Each window will span about 1,200 tokens before the onset and about 800 tokens of loop.

```python
# Hypothesis: C loops (gen_13 + gen_16 edited) give clean windows for a dense diagnostic TF pass
PREFIX='<|im_start|>user\n'; SUFFIX='\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n'
G13=[json.loads(l) for l in open('/work/gen_13.jsonl')]
print(len(G13), list(G13[0].keys()))
PROB={}
for fn in ['/practice/gsm8k_train.jsonl','/practice/math_train.jsonl']:
    for l in open(fn):
        d=json.loads(l); PROB[d['id']]=d
loopsC=[]
for gname,G in [('g13',G13),('g16',E16)]:
    for r in G:
        for s in r['samples']:
            if not s['cut']: continue
            t=s['text']; p=tail_period(t)
            if not p: continue
            on=loop_onset(t,p)
            loopsC.append(dict(g=gname,id=r['id'],src=r['source'],P=p,on=on,L=len(t),text=t))
print('C loops',len(loopsC), collections.Counter((x['g'],x['src']) for x in loopsC))
print('P quantiles',np.percentile([x['P'] for x in loopsC],[10,25,50,75,90]).round())
```