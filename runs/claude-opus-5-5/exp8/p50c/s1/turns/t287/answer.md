Repetition adds only about +0.7 logits of direct support for the copied token from the first copy to later cycles, and most of that comes from the layer-23 attention mixer (+0.4/+0.5). Every other component adds 0.3 or less. Even cycle 0, the original passage, already gets about 19 logits of support. So these loops are not driven by a runaway self-reinforcing component. The repeated content was highly predictable from the start, and the loop is a fixed point the model drifts into. Together with the earlier head-ablation results, this makes a single-component loop brake unlikely to exist under C. Before choosing between a damage/specificity check and some other mechanism, I want to see concretely how the remaining loops begin. GSM8K loops are the most puzzling, since the problems are easy (14% of GSM answers are still cut). For three GSM cut answers on H2 I will print short excerpts just before the exact-loop onset and the repeated unit. I will also print length quantiles of right, wrong and cut answers, to judge whether verbosity, rather than loops, limits the long correct answers.

```python
# Look at how GSM loops under C begin (excerpts before onset) + length distribution by outcome on H2 (C)
def char_loop(tx, maxp=3000):
    n=len(tx); tail=tx[-4000:]
    for p in range(5,maxp):
        if tail[-1500:]==tail[-1500-p:-p]: break
    else: return None
    i=n-1500-p
    while i-p>=0 and tx[i-1]==tx[i-1+p]: i-=1
    return p,i
toks={'ok':[],'wrong':[],'cut':[]}
ex=[]
for r in E27:
    for s in r['samples']:
        st_=status(s); toks[st_].append(s['tokens'])
        if st_=='cut' and r['source']=='gsm8k':
            cl=char_loop(s['text'])
            if cl: ex.append((r['id'],cl,s['text']))
for k,v in toks.items(): print(k, len(v), 'tok q25/50/75/90', np.percentile(v,[25,50,75,90]).astype(int))
print('gsm cut with exact char loop:', len(ex), 'of', sum(1 for r in E27 for s in r['samples'] if status(s)=='cut' and r['source']=='gsm8k'))
for idd,(p,i),tx in ex[:3]:
    print('=====', idd, 'period',p,'onset char',i,'of',len(tx))
    print('BEFORE:', repr(tx[max(0,i-500):i])[:520])
    print('UNIT:', repr(tx[i:i+min(p,250)]))
```