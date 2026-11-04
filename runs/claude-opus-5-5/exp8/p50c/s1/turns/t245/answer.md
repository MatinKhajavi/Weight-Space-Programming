The cross-tab shows head 23.4 has a dual role. Zeroing it rescues loops: cut→ok exceeds ok→cut by 15 (35 vs 20). It also breaks finished reasoning: ok→wrong exceeds wrong→ok by 15 (33 vs 18), mostly on GSM, where it goes 22 vs 14. The two effects cancel, so whole-head ablation of 23.4 is a dead end.

The other lead in my notes is the commit side. In about half of C's loops the correct answer had already been written before the loop began. The doubled brake column also pushes down `</think>` (about −0.5 logit at act −4) and "Ready" (−0.9) wherever it fires. That push is a side effect of my own edit and was not intended. Before spending a pass, I will check what C's successful endings look like: which line-initial words open the last line before `</think>`. If they are words the brake suppresses (Ready, Hmm, Actually…), removing those components from the added part of the column becomes a well-justified repair.

```python
# Hypothesis: C's closing ritual (last line(s) before </think>) uses words that the doubled brake column suppresses
def A(s): return s.encode('ascii','replace').decode()
recC = readj('/work/gen_13.jsonl') + readj('/work/gen_16_edited.jsonl')
ends, ends2 = [], []
for r in recC:
    s = r['samples'][0]; t = s['text']
    if s['correct'] and '</think>' in t:
        pre = t[:t.index('</think>')]
        lines = [l for l in pre.rstrip().split('\n') if l.strip()]
        ends.append(lines[-1].strip()[:70]); ends2.append(lines[-2].strip()[:50] if len(lines)>1 else '')
fw = lambda e: (e.split()[0] if e.split() else '')
print('n', len(ends), 'last-line first word:', collections.Counter(fw(e) for e in ends).most_common(15))
print('2nd-last first word:', collections.Counter(fw(e) for e in ends2).most_common(10))
for e in ends[:10]: print(repr(A(e)))
# how often do cut texts contain 'Ready' or ritual words at all
cuts = [r['samples'][0]['text'] for r in recC if r['samples'][0]['cut']]
oks = [r['samples'][0]['text'] for r in recC if r['samples'][0]['correct']]
for w in ['Ready', 'Final Answer', 'final answer', 'Final Polish', 'Final check', 'Self-Correction', 'Double check', 'Final Output']:
    print(w, 'cut %.2f ok %.2f' % (np.mean([w in t for t in cuts]), np.mean([w in t for t in oks])))
```