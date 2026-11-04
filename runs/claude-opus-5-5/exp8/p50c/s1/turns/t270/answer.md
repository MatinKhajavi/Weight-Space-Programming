X3 is clearly harmful: accuracy 49.4% vs 60.7% for C on the same 512 items, cut-offs up from 24.8% to 34.2%, wrong answers up from 75 to 85. Head 11.1 is not a loop driver. Removing it increases looping, so it serves general coherence, and I revert it now. That closes the whole-head ablation route. 23.4 was neutral, 11.1 is harmful, and the extra L23/L19 heads in edit B were harmful, so the remaining copying heads are load-bearing.

I return to a side effect of my own brake edit. Doubling the 22.573 column also suppresses "Ready", "Hmm", "Actually" and `</think>` where the neuron fires. "Ready…" lines belong to the closing ritual of successful answers (26% of correct texts vs 1% of cut texts). A paired comparison on the same S1 seeds tests this: A alone (no brake, gen_6_edited) vs C (gen_13). If C writes far fewer "Ready"/closing lines than A, the brake is partly blocking the model's own way of concluding, and protecting those tokens is a well-founded refinement.

```python
# Revert X3 (11.1). Hypothesis: the doubled brake also suppresses closing-ritual lines ('Ready ...') -> compare A (no brake) vs C on S1 seeds
drop_edit('model.language_model.layers.11.self_attn.o_proj.weight'); print(edited())
m25 = acc_map('/work/gen_25.jsonl'); k25=[k for k in m25 if k in mC]
d = np.array([m25[k][0]-mC[k][0] for k in k25], float); print('X3 vs C: %.1f SE %.1f, W->R %d R->W %d' % (100*d.mean(), 100*d.std(ddof=1)/np.sqrt(len(d)), (d>0).sum(), (d<0).sum()))
with open('/work/notes.md','a') as f: f.write("- PASS 20 = gen_25 (X3 = C + head 11.1 zeroed, SH512): acc 49.4 vs C 60.7 (diff %.1f SE %.1f), cut 34.2 vs 24.8 -> 11.1 needed (ablation INCREASES loops). Reverted to C.\n" % (100*d.mean(), 100*d.std(ddof=1)/np.sqrt(len(d))))
def line_rate(texts, pat):
    return np.mean([len(re.findall(r'(?m)^\s*(?:[*-]\s*)?(?:\*\*)?'+pat, t.split('</think>')[0]))>0 for t in texts])
for nm,p in (('A (gen_6_edited)','/work/gen_6_edited.jsonl'),('C (gen_13)','/work/gen_13.jsonl'),('brake-only (gen_15)','/work/gen_15.jsonl'),('orig (gen_6_original)','/work/gen_6_original.jsonl')):
    R=readj(p); T=[r['samples'][0]['text'] for r in R]; okT=[r['samples'][0]['text'] for r in R if r['samples'][0]['correct']]
    print(nm, 'acc %.2f' % np.mean([r['samples'][0]['correct'] for r in R]), '| share of ok texts with line: ' + ', '.join(f'{w} {line_rate(okT,w):.2f}' for w in ['Ready','Hmm','Actually','Final','Okay']), '| all texts Ready %.2f' % line_rate(T,'Ready'))
```