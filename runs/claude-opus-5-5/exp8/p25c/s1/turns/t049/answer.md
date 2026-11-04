About 60% of looping answers already contain the reference answer somewhere before the loop (72% GSM8K, 54% MATH, by a loose string match that overcounts small numbers). Explicit wrap-up statements are rare: "ready to" 3%, "I will write" 3%, a \boxed draft 19%, and </think> never. So the "sin 0°" case (post-solution rambling) is not typical. Most loops start mid-reasoning, typically in a "Wait, … / re-check" pattern, and then lock in. That keeps the exact verbatim loop, driven by the copy heads, as the main target. A fair share of looping problems are close to solved, so escaping loops should pay off. I will now build the dense measurement set:
- 40 looping answers with their real context up to loop start plus 6 periods.
- The 52 correct finished answers, as the benign reference.
- The 14 old compact texts, for consistency with fwd_4.

I save the token positions of every loop period. This pass will measure how each candidate edit changes (a) the chance that the evaluation's sampler leaves the loop and (b) the likelihood of normal correct reasoning.

```python
# Build TF measurement set: real-context loop texts (prefix+6 periods) + benign finished-correct answers + 14 compact
print('fin cols', fin.columns.tolist()); print(repr(fin_txt[0][:80]), '...', repr(fin_txt[0][-80:]))
print('fin_txt has </think>:', np.mean(['</think>' in t_ for t_ in fin_txt]))
cand = c2[(c2.onset_tok>=1200)&(c2.onset_tok<=5800)&(c2.period_tok>=8)&(c2.period_tok<=150)].copy()
cand = cand.sample(frac=1, random_state=7).drop_duplicates('id')
print('candidates distinct problems', len(cand), cand.src.value_counts().to_dict())
```