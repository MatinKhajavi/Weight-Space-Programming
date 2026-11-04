Under C, GSM8K answers are 70% correct, 14% cut and 16% wrong. MATH answers are 52% correct, 35% cut and 13% wrong. For GSM8K, 24 of the 36 cut answers had computed the correct number early (median 4% into the text) and nearly all (35) then fell into exact loops. The 16% wrong answers on easy grade-school problems are also surprisingly high for a thinking model, so I want to know what they are. Candidates are genuine reasoning errors, misreadings settled by a doubt spiral, or formatting and extraction problems (units, decimals, text inside \boxed{}). I will list predicted versus reference answers for C's wrong GSM8K answers and how long those answers ran.

```python
# What are C's wrong GSM answers? (pred vs ref, length) - reasoning errors vs formatting/extraction issues
wr = []
for name, g, ids in [('S1', g13, S1), ('H1', g16e, H1)]:
    for i in ids:
        s = g[i]
        if i.startswith('gsm') and str(s['correct']) != 'True' and str(s['cut']) != 'True':
            wr.append((i, str(s['pred'])[:40], str(ref[i]), int(s['tokens'])))
print(len(wr), 'median tokens', np.median([w[3] for w in wr]))
for w in wr[:41]:
    print(w[0][-5:], '| pred', repr(w[1]).encode('ascii', 'replace').decode(), '| ref', w[2], '| tok', w[3])
```