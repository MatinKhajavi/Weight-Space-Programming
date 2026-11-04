E5 beats E4b by a paired +7.81 ± 2.65 points (2.9 SE): MATH +8.1 ± 3.0, GSM8K +7.3 ± 5.0. Against the original the gain is +31.8 ± 2.6, with 175 w→r and only 12 r→w. The MATH gains now reach the harder levels (L3 0.38→0.49, L4 0.15→0.29). The share of finished answers that are correct keeps falling, though: 100% → 97.8% → 93.4% → 87.3%. Most new errors (24 of 31) are answers E4b never finished, but 5 were correct under E4b on the same seed, and GSM8K finished accuracy is only 87.8%. I want to know whether the edits make the model conclude on a wrong line of reasoning. One sign of that would be a shorter thinking phase than in correct answers; a second, final answers appearing right after the reasoning stops looping; a third, thinking that ends incoherently. The alternative is ordinary arithmetic or reading errors. So I compare thinking lengths of wrong vs correct finished answers and read short end-of-thinking excerpts of a few wrong GSM8K cases.

```python
# Inspect E5 wrong-finished answers: length vs correct finished, and short end-of-thinking excerpts (damage check)
fin14 = [k for k in G14 if not G14[k]['cut']]
wr14 = [k for k in fin14 if not G14[k]['correct']]
ok14 = [k for k in fin14 if G14[k]['correct']]
print('tokens: wrong-finished median', int(np.median([G14[k]['tokens'] for k in wr14])), ' correct-finished median', int(np.median([G14[k]['tokens'] for k in ok14])))
def think_end(t, n=260):
    j = t.find('</think>')
    if j == -1: return '[no </think>] ' + t[-n:]
    return t[max(0, j-n):j] + ' ||| ' + t[j+8:j+8+150]
for k in [x for x in wr14 if x[0].startswith('gsm')][:5]:
    r = G14[k]; ref = pid2prob[k[0]]['answer'] if isinstance(pid2prob[k[0]], dict) else ''
    print('---', k, 'tokens', r['tokens'], 'pred', r['pred'][:15], 'ref', str(ref)[:15])
    print(asc(think_end(r['text'])).replace('\n','|')[:420])
```