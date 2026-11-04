The ×4 brake has a measurable cost. On the 255 problems both versions finished, v4 lost 11 that v3 had right and gained only 4 (paired −2.7, SE 1.5).

The losses follow one pattern. v3 thought much longer on them (median 16.1k tokens against v4's 10.0k) and found the right answer. v4 stopped early with false confidence ("I am confident in this result" with 35 instead of 30). Accuracy among finished answers fell for both MATH (0.90 → 0.80) and GSM8K (0.84 → 0.75).

So eliminating "Wait" entirely (0.01 per 1,000 tokens) removes some genuine error-catching along with the loops. The net is still strongly positive, +9.4, because loop-breaking dominates.

GSM8K stands out: v4 reaches only 64%, and among finished GSM8K answers only 75% are correct. Easy grade-school problems should not fail that often once the model finishes. Before deciding whether to lower the dose or move to another target, I will read the GSM8K wrong answers. I will compare predicted and reference answers, check for grading or format issues (units, rounding), misinterpretation of the question, and early stopping without checking. That tells me whether GSM8K failures are a verification problem a softer brake would fix, or something else.

```python
# Hypothesis: v4's GSM8K wrong finishes are mostly misinterpretation/premature-confidence errors (not grading/format artifacts); inspect pred vs ref and thinking length
gw = [i for i in P5 if i.startswith('gsm') and not r11[i]['cut'] and not r11[i]['correct']]
ans = {rr['id']: rr['answer'] for rr in g11}
print('v4 GSM finished-wrong:', len(gw), '| v3 status on them:', collections.Counter(('cut' if r10[i]['cut'] else ('right' if r10[i]['correct'] else 'wrong')) for i in gw))
print('pred vs ref:', '; '.join(f"{str(r11[i].get('pred'))[:18]}|{ans[i]}" for i in gw[:24]))
print('median tokens of GSM wrong finishes', np.median([r11[i]['tokens'] for i in gw]), '| GSM right finishes', np.median([r11[i]['tokens'] for i in P5 if i.startswith('gsm') and r11[i]['correct']]))
gp = {rr_['id']: rr_ for rr_ in gs}
for i in gw[:3]:
    t = r11[i]['text']; k = t.find('</think>')
    print('--', i, 'ref', ans[i], 'pred', r11[i].get('pred'), '| Q:', gp[i]['problem'][:160].replace('\n', ' '), '| end:', t[max(0, k-220):k].replace('\n', ' | '))
```