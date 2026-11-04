34 of the 53 remaining cut-offs are exact loops (median period 108 characters). Their repeating units are now mostly deeply indented bullet lines with a computation ("*   $\binom{16}{5} = ...$"), not "Wait" lines. So line-level copying survives through some other route, now in list items. The loops start late (median at 43% of the text, around 14k tokens), after long struggles. Before deciding on a target, I will look at the 19 cut texts without an exact loop: near-loops with small variations, genuine enumerations, or whitespace blow-up. I will also measure how deep the indentation is in the loop lines.

```python
# Hypothesis: non-exact cut texts are near-loops (varying counters) or indentation blow-ups; inspect tails
for r in rows:
    if r[2] is None:
        pid=r[0]; s=[x[2] for x in cut if x[0]==pid][0]; t=s['text']
        tail=t[-400:]
        lines=t[-6000:].split('\n'); ind=[len(l)-len(l.lstrip(' ')) for l in lines if l.strip()]
        uniq=len(set(l.strip() for l in lines if l.strip()))/max(1,len([l for l in lines if l.strip()]))
        print(r[1][:4], pid[-12:], 'len',len(t),'tok',s['tokens'],'ind med',int(np.median(ind)),'uniq-line frac',round(uniq,2),'|',repr(tail[-220:]))
```