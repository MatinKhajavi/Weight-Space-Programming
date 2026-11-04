```python
# Hypothesis: loops start early-ish and persist to the cap; measure onset position (chars) of the final periodic tail.
def loop_onset(t, p):
    # walk back while t[i] == t[i+p]
    i = len(t)-p-1
    while i>=0 and t[i]==t[i+p]: i-=1
    return i+1
ons=[]; fracs=[]
for (rid,smp),p in zip(cut_rows,periods):
    if p:
        t=smp['text']; o=loop_onset(t,p); ons.append(o); fracs.append(o/len(t))
ons=np.array(ons); fracs=np.array(fracs)
print('onset frac of text: pcts 10/25/50/75/90', np.percentile(fracs,[10,25,50,75,90]).round(3))
print('onset chars pcts', np.percentile(ons,[10,25,50,75,90]).round(0))
nt = [ '</think>' in smp['text'] for rid,smp in cut_rows]
print('cut texts containing </think>:', sum(nt))
fin = [r for r in rows if not r['samples'][0]['cut']]
print('finished:', len(fin), 'correct among finished', sum(r['samples'][0]['correct'] for r in fin))
print('finished token lengths', sorted(r['samples'][0]['tokens'] for r in fin))
```