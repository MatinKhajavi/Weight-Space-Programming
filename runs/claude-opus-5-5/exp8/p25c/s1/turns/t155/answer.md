The half-strength edit barely works: 14.1% over H + H2 (GSM8K 18.8%, MATH 11.3%), with 85% of answers still cut. The original scored 11.5% on the same problems, E6 42.6% and E6-strong 43.2%. The dose-response is therefore threshold-like. At scale 0.5 (attention output inside the subspace multiplied by −0.5 at L15/L19, merely removed at L23) loops still lock in. This fits the compensation seen earlier (about 70% compensation for the rank-1, β = 1 edit E2, with the L22 MLP relaxing): the reversal must exceed the compensating paths before the copied opener loses. Between scale 1.0 and 1.5 accuracy is flat, so the knee lies somewhere between 0.5 and 1.0. Next I will compute the paired statistics for half strength. I will also check whether E6-strong's higher count of finished answers without \boxed (10 vs 2 on H + H2) shows harm from the stronger edit, which matters for choosing E6 or E6-strong. A scale-0.75 run would then locate the knee and show how far E6 sits from it.

```python
# Dose-response: paired stats for E6-half on H+H2; inspect E6-strong 'finished but no answer' cases for formatting damage.
G[22]=readj('/work/gen_22.jsonl'); G['22H']=[r for r in G[22] if r['id'] in set(H)]; G['22H2']=[r for r in G[22] if r['id'] in set(H2)]
for src in [None,'gsm8k','math']:
    print(src, 'half-orig', pooled([(G['18o'],G['22H']),(G['21o'],G['22H2'])], src), '| E6-half', pooled([(G['22H'],G['18e']),(G['22H2'],G['21e'])], src),
          '| E6s-orig', pooled([(G['18o'],G['20H']),(G['21o'],G['20H2'])], src))
acc_=lambda gl: np.mean([s['correct'] for g_ in gl for r in g_ for s in r['samples']])
cut_=lambda gl: np.mean([s['cut'] for g_ in gl for r in g_ for s in r['samples']])
for lab,gl in [('s=0 orig',[G['18o'],G['21o']]),('s=0.5',[G['22H'],G['22H2']]),('s=1 E6',[G['18e'],G['21e']]),('s=1.5',[G['20H'],G['20H2']])]:
    print('%-9s acc %.3f cut %.3f'%(lab, acc_(gl), cut_(gl)))
noans=[(r,s) for r in G[20] for s in r['samples'] if (not s['cut']) and (s['pred'] in (None,'',[]))]
print('E6s finished-no-answer:', len(noans))
for r,s in noans[:4]: print(lvl_key(r), 'tok', s['tokens'], '| end:', A_(s['text'][-200:]).replace('\n','\\n'))
```