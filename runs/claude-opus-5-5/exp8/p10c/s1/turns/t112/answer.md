The GSM8K wrong answers in v4 are genuine reasoning errors, not grading artifacts. The model misreads the question or assumes a missing quantity and stops:
- In the baseball problem it reads "4 times for 4 weeks" as 16 weeks and answers 4 instead of 1.
- In the TV problem it assumes a day is 100 minutes and answers 20 instead of 288.

Of the 27 GSM8K wrong finishes, v3 was also wrong on 11, cut on 10 and right on only 6. So the verification cost of the ×4 brake is real but small next to its loop-breaking gain (+9.4 net). Lowering the dose would trade back some of that gain for a point or so of verification.

The larger remaining category is still answers cut at the limit: 140 of 512, of which 91 are exact loops and only 35% doubt-laden now. The rest look like restatement or recomputation loops ("So …", "If …", formulas). Their lock-in is written by the copy heads, which are only half-suppressed.

Before deciding whether pass 7 goes to a stronger copy-head dose, I will read short excerpts of v4's remaining loops and non-exact cuts and check their periods and onsets. Mostly short exact restatement or recomputation cycles would make the copy circuit the right target. Mostly long legitimate enumerations would make further loop edits pointless, and I would spend the passes on dose-response and confirmation instead.

```python
# Hypothesis: v4's remaining cut answers are mostly short doubt-free restatement/recomputation loops (copy lock-in), not long legitimate enumerations
ex4 = []; nonex4 = []
for i in P5:
    s = r11[i]
    if s['cut']:
        ids = toks(s['text']); on, p = tail_loop(ids)
        if on is not None: ex4.append((i, on, p))
        else: nonex4.append((i, ngram_rep(ids)))
ps = np.array([p for _, _, p in ex4]); ons = np.array([o for _, o, _ in ex4])
print(f"v4 exact loops {len(ex4)}: period median {np.median(ps):.0f} (10-90%: {np.percentile(ps,10):.0f}-{np.percentile(ps,90):.0f}) | onset 10/50/90%: {np.percentile(ons,10):.0f}/{np.median(ons):.0f}/{np.percentile(ons,90):.0f}")
rp = np.array([r for _, r in nonex4]); print(f"v4 non-exact cuts {len(nonex4)}: repeated-20gram share median {np.median(rp):.2f}, >0.5: {(rp>0.5).sum()}, <0.1: {(rp<0.1).sum()}")
random.seed(3)
for i, on, p in random.sample(ex4, 5):
    ids = toks(r11[i]['text']); print('-- loop', i, 'p', p, 'on', on, '|', tok.decode(ids[-p:])[:230].replace('\n', ' | '))
for i, r_ in random.sample(nonex4, 3):
    print('-- nonex', i, f'rep {r_:.2f} |', r11[i]['text'][-230:].replace('\n', ' | '))
```