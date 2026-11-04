About 36% of E6 cut-offs (178 of 496) have no exact loop, but nearly all of these are near-loops (median near-repetition 0.97). They cycle through slightly varying lines, often wrap-up statements ("So the answer is 120. / I'll put 120 in the box. / This is the answer.") or bullet restatements. A few are counting enumerations. The other 64% are short exact loops (median period 57 characters). So the remaining failure is still repetition. The opener-subspace approach is saturated: E7 caused loops to migrate and the dose-response plateaus.

The family-agnostic driver is the copy machinery itself. My earlier measurements found layer 23 attention to be the largest contributor to the growth of the loop token's logit (+2.5 of +3.4), with heads 5 (induction) and 6 (same-token) rising most. Ablating them (E1) tripled per-period escape in teacher forcing but gave only about +2 points alone, plausibly because the "Wait" opener lock (L15/L19) was still intact. Before proposing E6 plus head dampening, I'll check whether E1 damaged reasoning: its finished-answer accuracy, and its per-problem effect on GSM8K versus MATH. I'll also re-read the notes junction about E1.

```python
# Check E1 (L23 h5,h6 ablated, gen_8) for reasoning damage: finished-sample accuracy and paired-by-source vs original gen_2
print(notes[2300:2800])
def S(fn): 
    rows=[json.loads(l) for l in open('/work/'+fn)]; return {r['id']:(r['source'],r['samples']) for r in rows}
g2=S('gen_2.jsonl'); g8=S('gen_8.jsonl')
for nm,g in [('orig',g2),('E1',g8)]:
    fin=[s for src,ss in g.values() for s in ss if not s['cut']]
    print(nm,'finished',len(fin),'fin-acc',round(np.mean([s['correct'] for s in fin]),3),'acc',round(np.mean([s['correct'] for src,ss in g.values() for s in ss]),3))
both_fin=[(a['correct'],b['correct']) for i in g2 for a,b in zip(g2[i][1],g8[i][1]) if not a['cut'] and not b['cut']]
print('both finished', len(both_fin), 'acc orig/E1', np.mean([x[0] for x in both_fin]).round(3), np.mean([x[1] for x in both_fin]).round(3))
```