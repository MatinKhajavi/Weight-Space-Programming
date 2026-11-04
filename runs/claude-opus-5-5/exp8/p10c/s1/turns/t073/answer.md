Ablating the neuron cut the model's use of "Wait" about fourfold: ' Wait', 'Wait' and ' wait' together went from about 7.9 to 2.1 per 1,000 tokens. Part of this shifted to "But" (0.8 → 1.9 per 1,000 tokens). Loops now start later (median 9.0k instead of 6.0k tokens) and have longer cycles (65 instead of 42 tokens). But 143 of the 170 remaining cut-off answers are still exact periodic loops, and only about a third of their cycles contain a "Wait". So the doubt neuron explains much of the loop *formation*, but a second mechanism keeps the model locked once any paragraph repeats. That points back to the copying heads, or to doubt expressed with other words. To decide which comes next, I look at what the remaining loop cycles contain: short excerpts of a few cycles, how many cycles have no doubt word at all, and the most common first word of the lines in a cycle. I also check whether the reference answer already appears before loop onset, which tells me how much accuracy is still recoverable by breaking the remaining loops. If most cycles have no doubt word, the lock-in comes from generic copying of repeated text, and damping the strongest copying heads is the next causal test.

```python
# What do remaining loops (edited) contain? excerpts, doubt-free share, line-start words; answer present before onset?
def ans_before(text_ids, on, ans):
    pre_txt = tok.decode(text_ids[:on]); a = str(ans).strip().replace('\\dfrac','\\frac')
    return a in pre_txt.replace('\\dfrac','\\frac')
rng2 = np.random.default_rng(1)
for m in ('original','edited'):
    rows = R8[m]; loops = [(r, g) for r, g in zip(rows, G8[m]) if r['cut'] and r['on'] is not None]
    nod = np.mean([not any(r['cyc'][w] for w in DOUBT) for r,_ in loops])
    ab = np.mean([ans_before(tok.encode(g['samples'][0]['text']).ids, r['on'], g['answer']) for r, g in loops])
    firsts = Counter()
    for r, g in loops:
        ids = tok.encode(g['samples'][0]['text']).ids; cyc_txt = tok.decode(ids[r['on']:r['on']+r['p']])
        for ln in cyc_txt.split('\n')[1:]:
            w = ln.strip(' *-0123456789.').split(' ')[0][:10]
            if w.isalpha(): firsts[w] += 1
    print(f'{m}: loops {len(loops)} | cycles with no doubt word {nod:.2f} | ref answer string before onset {ab:.2f}')
    print('   common first words of lines in cycles:', [(w,c) for w,c in firsts.most_common(14) if w.isascii()])
loops_e = [(r, g) for r, g in zip(R8['edited'], G8['edited']) if r['cut'] and r['on'] is not None]
for k in rng2.choice(len(loops_e), 3, replace=False):
    r, g = loops_e[k]; ids = tok.encode(g['samples'][0]['text']).ids
    print('---', r['id'], 'onset', r['on'], 'p', r['p'], '|', repr(safe(tok.decode(ids[r['on']:r['on']+r['p']]))[:240]))
```