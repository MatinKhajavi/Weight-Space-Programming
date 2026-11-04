The gen rows don't store the problem text, so I'll pull problems from the practice files by id. Then I'll assemble the exact evaluation-format texts. For each text I'll record the token index where the answer starts, the loop onset, and the cycle boundaries (from character offsets, using the same tokenizer). I'll check the total against the 524,288-token limit and save the corpus to /work so it can be reused for every configuration.

```python
# Hypothesis: none (setup) - assemble exact-format TF texts with token-level region boundaries, save corpus
PRB={}
for fn in ['gsm8k_train','math_train','aime']:
    for l in open(f'/practice/{fn}.jsonl'): o=json.loads(l); PRB[o['id']]=o
TF=[]; 
for rid,kind,ans,on,P in corpus:
    pr=PREFIX+PRB[rid]['problem']+SUFFIX; full=pr+ans
    enc=tok.encode(full); offs=enc.offsets; n=len(enc.ids)
    def c2t(c):  # first token index whose start >= c
        return next((i for i,o in enumerate(offs) if o[0]>=c), n)
    a0=c2t(len(pr))
    rec={'id':rid,'kind':kind,'n':n,'a0':a0}
    if kind=='loop':
        rec['on']=c2t(len(pr)+on); rec['cyc']=[c2t(len(pr)+on+k*P) for k in range(7)]
    TF.append((full,rec))
ntot=sum(r['n'] for _,r in TF); print('texts',len(TF),'tokens',ntot,'max',max(r['n'] for _,r in TF))
print('loop onset tok pct',np.percentile([r['on']-r['a0'] for _,r in TF if r['kind']=='loop'],[10,50,90]).astype(int),
      'cycle len tok median',int(np.median([r['cyc'][1]-r['cyc'][0] for _,r in TF if r['kind']=='loop'])))
json.dump({'texts':[t for t,_ in TF],'recs':[r for _,r in TF]},open('/work/tf_corpus1.json','w'))
```