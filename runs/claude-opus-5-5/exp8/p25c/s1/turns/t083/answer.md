L11loop holds 350 looping E3 answers with period P (in characters) and onset; the categories are verification 146, other 142, wrap-up 47 and degenerate 15. Next I will read the actual loop periods: about 3 short excerpts per category, at most 300 characters each, shown as ASCII. I will also count line starts per period and the openers they use across all 350 loops. If most periods contain at least one newline followed by an opener word, the line-start decision-point mechanism (the one E3 broke for Wait) is still the bottleneck.

```python
# Hypothesis check (cont.): structure of E3 loop periods -- line starts per period and opener words
k0 = list(G11.keys())[0]; print(type(G11[k0]), list(G11[k0].keys())[:12] if isinstance(G11[k0], dict) else '')
def get_text(rec):
    for kk in ('text','output','completion','response'):
        if kk in rec: return rec[kk]
    return None
def asc(s): return s.encode('ascii','replace').decode()
per_stats = []; open_ctr = collections.Counter(); nls = []
for _, row in L11loop.iterrows():
    rec = G11[row['key']] if row['key'] in G11 else G11[tuple(row['key'])]
    t = get_text(rec); P = int(row['P'])
    period = t[-P:]
    # count line starts inside one period (newline followed by a word)
    starts = re.findall(r'\n+\s*([A-Za-z\*\-\$\\]+)', period + period[:40])
    nls.append(len(re.findall(r'\n', period)))
    for w in starts: open_ctr[w] += 1
    per_stats.append((row['key'], P, len(starts)))
print('newlines per period: share 0 =', np.mean(np.array(nls)==0), ' median', np.median(nls))
print('top openers:', [(w,c) for w,c in open_ctr.most_common(25) if w.isascii()][:25])
for cat_name in ['verification','other','wrap-up']:
    pass
rng4 = np.random.RandomState(7)
sel_rows = L11loop.sample(6, random_state=3)
for _, row in sel_rows.iterrows():
    rec = G11[row['key']]; t = get_text(rec); P = int(row['P'])
    print('---', row['key'], 'P', P, 'onset', row['onset_tok'])
    print(asc(t[-P:][:300]).replace('\n','|'))
```