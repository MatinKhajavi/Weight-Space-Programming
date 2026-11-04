Wrap-up phrases ("Ready to write", "Final check", "I'm confident") are common in finished answers (40-80%) but rare in looping ones (2-11%), and I found no instance of "Done thinking" being followed by further thinking. So looping answers do not reach a closing point and then fail to emit </think>; they fall into the loop well before the wrap-up stage. The failure to investigate is therefore the repetition loop itself. My plan is a content-controlled comparison: in each looping text, the same token sequence occurs once before the loop locks in and then repeats many times, so comparing activations on identical tokens at the 1st versus later repetitions isolates a "repetition state" independent of content. To keep teacher-forced texts short (activations are returned for every position), I'll first find looping samples whose loop starts early, and compute loop period and onset in tokens. If I find enough samples with onset under ~3,000 tokens, I'll use them for a teacher-forced pass returning per-token log-probabilities, top-k alternatives, residual streams at all layers and MLP activations.

```python
# Find early-onset loops (in tokens) for teacher-forced analysis
c2 = c2.copy()
c2['onset_tok'] = [ntok(t[:int(st)]) for t, st in zip(c2.text, c2.loopstart)]
c2['period_tok'] = [ntok(t[int(st):int(st)+int(p)]) for t, st, p in zip(c2.text, c2.loopstart, c2.period)]
print(c2.onset_tok.describe().round(0).to_dict())
early = c2[(c2.onset_tok < 2600) & (c2.period_tok >= 12) & (c2.period_tok <= 80)]
print(len(early), early.groupby('src').size().to_dict())
print(early[['id','onset_tok','period_tok']].head(20).to_string())
```