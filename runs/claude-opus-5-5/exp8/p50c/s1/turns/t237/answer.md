The loop-detector idea fails: these heads use no attention sink, and their induction mass is already high in ordinary text. Before exact loops it is 0.27–0.39 in the original's windows, with 22–38% of positions above 0.5, and 0.6–0.68 in the near-repetitive stretches of the edited models' windows. An exit circuit driven by their attention would fire constantly during normal reasoning, so I am dropping it. Within C, the most fixable-looking failure remains GSM8K cut-offs: 35 of 36 are exact loops, and in 24 the correct number was computed very early (median 4% into the text). The model solves an easy problem and then spirals. I want to see the transition: what the model writes right after first computing the answer in these failing cases. If a common trigger (for example a "check the wording / ambiguity / trick" ritual) starts the spiral, that would be a specific behaviour to trace.

```python
# What happens right after the GSM answer is first computed in C's looping GSM cases? (short excerpts at the transition)
gsm_loop = [i for i in ids512 if i.startswith('gsm') and st(gC[i]) == 'cut']
random.seed(3); shown = 0
for i in random.sample(gsm_loop, len(gsm_loop)):
    t = gC[i]['text']; a = str(ref[i]).replace(',', '')
    m = re.search(rf'=\s*\$?{re.escape(a)}(?![0-9])', t.replace(',', ''))
    if not m: continue
    j = m.end(); print('-', i[-5:], 'ans', a, '@', round(j / len(t), 3), '|', repr(t[j:j + 330]).encode('ascii', 'replace').decode()); shown += 1
    if shown >= 6: break
words = ['wording', 'ambigu', 'interpret', 'trick', 'phrasing', 're-read', 'literal', 'Is it possible', 'refers to']
for tag, cond in [('gsm ok', lambda i: st(gC[i]) == 'ok'), ('gsm cut', lambda i: st(gC[i]) == 'cut'), ('gsm wrong', lambda i: st(gC[i]) == 'wrong')]:
    sel = [i for i in ids512 if i.startswith('gsm') and cond(i)]; ntk = sum(int(gC[i]['tokens']) for i in sel) / 1000
    print(tag, len(sel), {w: round(sum(gC[i]['text'].count(w) for i in sel) / ntk, 2) for w in words})
```