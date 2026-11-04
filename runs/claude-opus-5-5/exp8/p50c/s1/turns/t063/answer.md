The result clearly confirms the hypothesis. Ablating the three loop-locking induction heads raised accuracy on S1 from 12.5% to 35.2% (paired difference +22.7 points, SE 2.7), with 60 samples going wrong to right and only 2 right to wrong. Cut-offs fell from 89% to 61%, and GSM8K and MATH both improved (15.6→43.0%, 9.4→27.3%). Still, 61% of edited answers are cut off. I'll now check whether those are still verbatim loops (and with what periods and onsets) or a different failure. That decides whether to ablate more copy heads, for example 23.7, 23.4, 19.5, 19.7 and 11.1, or to look elsewhere. I'll log the result first and keep a copy of edit A.

```python
# Hypothesis: remaining cut-offs in edit A are still verbatim loops (driven by remaining copy heads), now starting later
import shutil
with open('/work/notes.md','a') as f: f.write("""
- PASS 3 = gen_6 (both, S1, n=1, 32768): ORIGINAL acc 12.5% (gsm 15.6, math 9.4), cut 89.5% | EDIT A acc 35.2% (gsm 43.0, math 27.3), cut 61.3%. Paired diff +22.7 pts SE 2.7; W->R 60, R->W 2. ==> induction-head ablation strongly reduces loops and helps.
""")
if os.path.exists('/work/edited'): shutil.copytree('/work/edited','/work/best_A',dirs_exist_ok=True)
def summarize(fn):
    R=[json.loads(l) for l in open(fn)]; out=[]
    for r in R:
        for s in r['samples']:
            on,P=find_loop(s['text']) if s['cut'] else (None,None)
            out.append(dict(id=r['id'],src=r['source'],c=bool(s['correct']),cut=bool(s['cut']),tok=int(s['tokens']),on=on,P=P))
    return pd.DataFrame(out)
import pandas as pd
g6o=summarize('/work/gen_6_original.jsonl'); g6e=summarize('/work/gen_6_edited.jsonl')
for nm,g in [('orig',g6o),('editA',g6e)]:
    c=g[g.cut]; print(nm,'cut',len(c),'exact loop',c.on.notna().sum(),'onset char med',c.on.median(),'P med',c.P.median(),'correct tok med',g[g.c].tok.median())
```