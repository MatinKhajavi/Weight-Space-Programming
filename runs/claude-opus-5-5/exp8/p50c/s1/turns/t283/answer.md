That output was wasted: the helpers can't show their source because they were defined through exec, and a sample text filled the output. I did learn the record format: id, source, answer, and a samples list holding a text and grading fields. Next I want to know whether whole-head ablations failed because they damaged useful computation or because breaking loops just exposes problems the model cannot solve. For 23.4 (X1), 11.1 (X3) and 23.7 (X4), I will cross-tabulate each answer's outcome under C (right / wrong / cut) against its outcome under each ablation, using the shared random seeds. Many "right under C → wrong but finished" transitions would point to damage. If so, an edit that removes only the heads' line-start retrieval could keep the loop reduction without the cost. I will also list some of C's wrong answers on H2, to check whether part of the "wrong" category is just answer formatting.

```python
# Q: did whole-head ablations (23.4, 11.1, 23.7) fail through damage (C-ok -> finished-wrong) or only by exposing unsolvable loops? + H2 wrong-answer formats
import json, collections
def lg(p): return [json.loads(l) for l in open(p)]
E27 = lg('/work/gen_27_edited.jsonl')
s0 = E27[0]['samples'][0]
print({k:(v if not isinstance(v,str) else v[:40]) for k,v in s0.items() if k!='text'})
def status(s):
    g=None
    for k in ('correct','is_correct','grade','ok','score'):
        if k in s: g=s[k]; break
    if isinstance(g,str): g=g.lower() in ('true','correct','1','right')
    if g: return 'ok'
    f=str(s.get('failure',s.get('finish_reason',''))).lower()
    for k in ('cut','truncated'):
        if k in s: return 'cut' if s[k] else 'wrong'
    return 'cut' if ('cut' in f or 'length' in f or 'max' in f) else 'wrong'
def smap(paths):
    m={}
    for p in paths:
        for r in lg(p):
            for i,s in enumerate(r['samples']): m[(r['id'],i)]=(status(s),r['source'])
    return m
C = smap(['/work/gen_13.jsonl','/work/gen_16_edited.jsonl'])
for name,p in [('X1_23.4','/work/gen_22.jsonl'),('X3_11.1','/work/gen_25.jsonl'),('X4_23.7','/work/gen_26.jsonl')]:
    X = smap([p]); keys=[k for k in X if k in C]
    ct = collections.Counter((C[k][0],X[k][0]) for k in keys)
    print(name, len(keys), {f'{a}->{b}':ct[(a,b)] for a in ('ok','cut','wrong') for b in ('ok','cut','wrong')})
st = collections.Counter(status(s) for r in E27 for s in r['samples']); print('H2 C', st)
rows=[]
for r in E27:
    for s in r['samples']:
        if status(s)=='wrong':
            ex = s.get('extracted', s.get('pred', s.get('answer_extracted','?')))
            rows.append((r['source'][:4], str(r['answer'])[:22], str(ex)[:22]))
for x in rows[:30]: print(x)
```