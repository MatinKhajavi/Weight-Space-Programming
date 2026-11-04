Against a control of other problems' answers (31.3%), the excess of the answer appearing before loop onset (55.9%) is real but moderate. About a third of the cut answers had effectively solved the problem and then failed to conclude. The rest were still working when they looped or hit the cap. The non-exact cut texts show little near-repetition (tail line reuse 0.23 vs 0.18 for long correct texts), so they are simply very long reasoning. Correct C answers end with an output-planning statement such as "Ready to generate output." followed by </think>. Next I check whether the head ablation costs accuracy on problems where the model does finish. Brake-only had 8 finished-but-wrong answers on S1 versus 28 for A+brake. I will cross-tabulate per-problem outcomes of brake-only (gen_15) against A+brake (gen_13) on the same seeds. If many brake-only correct answers become wrong (not cut) under A+brake, the head ablation also damages reasoning, and a weaker or more selective ablation could recover accuracy. If the extra wrong answers come mostly from problems that were cut under brake-only, they are just hard problems now finishing.

```python
# Hypothesis: head ablation A (on top of brake) converts some correct answers into wrong ones (damage), not only cut->finished; cross-tab gen_15 (brake-only) vs gen_13 (A+brake) on S1, same seeds
g15=loadj('/work/gen_15.jsonl'); g6e=loadj('/work/gen_6_edited.jsonl'); g6o=loadj('/work/gen_6_original.jsonl'); g14=loadj('/work/gen_14.jsonl')
def stmap(G_): return {r_['id']:('ok' if r_['samples'][0]['correct'] else ('cut' if r_['samples'][0]['cut'] else 'wrong')) for r_ in G_}
mB=stmap(g15); mC=stmap(gC13); mA=stmap(g6e); mO=stmap(g6o); m15x=stmap(g14)
ct=pd.crosstab(pd.Series([mB[i] for i in mC],name='brake_only'), pd.Series([mC[i] for i in mC],name='A+brake'))
print(ct)
ct2=pd.crosstab(pd.Series([mA[i] for i in mC],name='A_only'), pd.Series([mC[i] for i in mC],name='A+brake'))
print(ct2)
# by source for wrong counts
for nm_r, mm_ in [('orig',mO),('A',mA),('brake',mB),('A+brake1.5',m15x),('A+brake2',mC)]:
    v_=collections.Counter((('gsm' if 'gsm' in i else 'math'), s) for i,s in mm_.items())
    print(nm_r, {k:v_[k] for k in sorted(v_)})
```