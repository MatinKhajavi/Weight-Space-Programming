No neuron stands out as a broad doubt promoter at line starts: the largest pushes along the doubt-minus-progress direction are only 0.03–0.04 per unit (about 0.15 logits after the final norm), from 20.1111 and the previously known 16.998. So the doubt impulse is not concentrated in a few MLP neurons. Earlier evidence says it comes mainly through attention re-reading the model's own earlier lines, which edits X1 and B showed cannot be cut further without harming correct work. The token-level, head-level and neuron-level routes against loops now all look exhausted around C. Before choosing among the remaining options, I want one more qualitative check on C's wrong GSM8K answers. Many are off by exactly a factor of 2. A systematic misreading pattern (such as doubling for "each"/"pairs"/"round trip") would be a different failure mechanism from looping; random errors would not be.

```python
# Do C's factor-of-2 GSM errors share a systematic misreading? Show problem + the model's final computation line (short excerpts)
gref = {json.loads(l)['id']: json.loads(l) for l in open('/practice/gsm8k_train.jsonl')}
f2 = []
for i in [w[0] for w in wr]:
    s = gC[i]; p = re.sub(r'[^0-9.]', '', str(s['pred']));
    try:
        r_, p_ = float(ref[i]), float(p)
        if r_ and p_ and (abs(p_ / r_ - 2) < 1e-6 or abs(r_ / p_ - 2) < 1e-6): f2.append(i)
    except Exception: pass
print('factor-2 errors', len(f2), 'of', len(wr))
for i in f2[:5]:
    t = gC[i]['text']; k = t.find('</think>')
    print('-', i[-5:], 'pred', gC[i]['pred'], 'ref', ref[i], '| Q:', gref[i]['problem'][:230].encode('ascii', 'replace').decode())
    print('   final:', repr(t[k + 9:k + 260] if k > 0 else t[-250:]).encode('ascii', 'replace').decode())
```