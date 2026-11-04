Under E4b, Let, Okay and If have disappeared from the loops entirely, so the suppression works for them. "So" survives and now appears in 36% of loops (up from 20% under E3); 46% of loops contain only bullet (`*`) line starts, and the median period shrank from 153 to 67 characters. Two things may explain "So": a stronger push from layers other than L15/L19, or the centred reversal of the other openers redistributing probability toward So. Before choosing the next lever, I want to see what the two dominant loop types look like. For bullet loops: what follows the `*` (the real decision point may be the first word after the bullet). For So loops: is it a one-line restatement such as "So the answer is X."? I will print short excerpts, at most 250 characters each, of several of each, and tabulate the first word after `*   ` in bullet-only periods.

```python
# Inspect E4b's dominant remaining loop types: bullet-only loops (what follows '*') and So-loops
looped12 = [x for x in LT12 if x[1] is not None]
bul = [x for x in looped12 if len(x[3])>0 and set(x[3]) <= {'*','-'}]
sol = [x for x in looped12 if 'So' in x[3]]
after_bul = collections.Counter()
for x in bul:
    per = G12[x[0]]['text'][-x[1]:]
    for w in re.findall(r'\n\s*[\*\-]\s+\**([A-Za-z\$\\]+)', per + per[:80]):
        after_bul[w] += 1
print('first word after bullet (bullet-only loops):', [(w,c) for w,c in after_bul.most_common(20) if w.isascii()])
rs = np.random.RandomState(11)
for x in [bul[i] for i in rs.choice(len(bul), 4, replace=False)]:
    print('--- BUL', x[0], 'P', x[1], 'onset_char', x[2], 'len', len(G12[x[0]]['text']))
    print(asc(G12[x[0]]['text'][-x[1]:][:250]).replace('\n','|'))
for x in [sol[i] for i in rs.choice(len(sol), 4, replace=False)]:
    print('--- SO', x[0], 'P', x[1])
    print(asc(G12[x[0]]['text'][-x[1]:][:250]).replace('\n','|'))
```