With the decision positions fixed, I can see what pushes " Wait" at line starts where the model is about to doubt. Relative to other line starts, the Wait logit is 3.3 higher there:

- **Neuron 998** (layer 16, already ablated) supplies +0.98, essentially all of layer 16's MLP push.
- **Attention layers** supply most of the rest: layer 22 DeltaNet +0.92, layer 15 full attention +0.52, layer 21 DeltaNet +0.48, layer 23 full attention +0.30, layer 20 DeltaNet +0.22.
- **Other MLPs** add smaller amounts: layer 21 +0.52, layer 23 +0.46, layer 18 +0.32.
- **Brake:** layer 22's MLP is a strong natural brake against Wait (−1.97).

So, apart from neuron 998, doubt is carried by attention reading earlier doubt from the context. The late copy layers matter for both lock-in and doubt, but not overwhelmingly for doubt.

Before choosing between more copy-head suppression and something aimed at re-checking, I need the timeline of failures in v2. The question is whether loopers find the answer and then spend thousands of tokens re-checking before the exact loop starts, or lock in soon after finding it. I will locate the first appearance of the reference answer in each v2 loop text and compare the gap to loop onset with finishers' time from first answer to </think>. A long gap means over-verification is the main problem and the exact loop is just the terminal state. A short gap means the lock-in itself is stealing samples that would otherwise finish.

```python
# Hypothesis: v2 loopers usually find the answer long before the exact loop starts (over-verification), vs finishers who close soon after first finding it
think_end = 248069
def first_ans_tok(text, ans):
    a = str(ans).strip()
    if not a: return None
    pat = (r'(?<![\d.])' + re.escape(a) + r'(?![\d])') if re.fullmatch(r'-?[\d.,/]+', a) else re.escape(a)
    m = re.search(pat, text)
    if not m: return None
    e = tok.encode(text); offs = e.offsets
    return next((k for k, (s0, s1) in enumerate(offs) if s1 > m.start()), None)
gaps_loop, gaps_fin, fin_len, pre_found = [], [], [], 0
for rr in g9e:
    s = rr['samples'][0]; text = s['text']
    if s['cut']:
        ids = toks(text); on, p = tail_loop(ids)
        if on is None: continue
        fa = first_ans_tok(text, rr['answer'])
        if fa is not None and fa < on: gaps_loop.append((fa, on)); pre_found += 1
    else:
        ids = toks(text); fin_len.append(len(ids))
        if think_end in ids:
            te = ids.index(think_end); fa = first_ans_tok(text, rr['answer'])
            if fa is not None and fa < te and s['correct']: gaps_fin.append((fa, te))
gl = np.array(gaps_loop); gf = np.array(gaps_fin)
print(f"v2 loops with ref answer before onset: {pre_found}/144 | first-answer tok median {np.median(gl[:,0]):.0f}, onset median {np.median(gl[:,1]):.0f}, gap median {np.median(gl[:,1]-gl[:,0]):.0f} (10-90%: {np.percentile(gl[:,1]-gl[:,0],10):.0f}-{np.percentile(gl[:,1]-gl[:,0],90):.0f})")
print(f"v2 finished correct: n {len(gf)} | first-answer median {np.median(gf[:,0]):.0f}, </think> median {np.median(gf[:,1]):.0f}, gap median {np.median(gf[:,1]-gf[:,0]):.0f} (10-90%: {np.percentile(gf[:,1]-gf[:,0],10):.0f}-{np.percentile(gf[:,1]-gf[:,0],90):.0f})")
print(f"v2 finished total length: median {np.median(fin_len):.0f}, 90% {np.percentile(fin_len,90):.0f}, max {max(fin_len)}")
```