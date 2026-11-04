The screening pass worked: 512 openings of 12 tokens each under C. Now I count the opening styles, overall and by source, and find the problems whose two samples open differently, one "my thought process" and one "a thinking process". Those discordant problems are the units of the randomized within-problem test. I also look at the first-token split, "Here's" vs "Here is", for the later mechanism work. Seeing about 20% "my thought" openings with a reasonable number of discordant problems would confirm the design is feasible.

```python
# Screening analysis: opening styles per sample; problems with discordant openings (one 'my_thought', one 'a_thinking')
R23 = readj('/work/gen_23.jsonl')
st23 = {r['id']: [sty(s['text'].lstrip()[:28]) for s in r['samples']] for r in R23}
src23 = {r['id']: r['source'] for r in R23}
allst = collections.Counter(x for v in st23.values() for x in v)
print('styles:', dict(allst))
for src in ('gsm8k','math'):
    c = collections.Counter(x for i,v in st23.items() if src23[i]==src for x in v); print(src, dict(c))
disc = [i for i,v in st23.items() if set(v)=={'my_thought','a_thinking'}]
both_my = [i for i,v in st23.items() if v==['my_thought','my_thought']]
print('discordant my/a:', len(disc), {s: sum(src23[i]==s for i in disc) for s in ('gsm8k','math')}, '| both my:', len(both_my))
print('first two tokens:', collections.Counter(A(s['text'].lstrip()[:9]) for r in R23 for s in r['samples']).most_common(6))
json.dump(disc, open('/work/O_disc_ids.json','w'))
```