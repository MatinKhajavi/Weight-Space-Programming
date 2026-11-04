The wrong finished answers are real reasoning errors, not format problems; only 1-2 are unit or format quirks such as '\$120' or a tuple. They are also very long (median about 19k tokens even for GSM8K), so the model struggles for a long time before committing. Next I will check the 52 cut-offs: how many end in exact periodic loops, the period and onset, and what the repeating units start with now that line-initial "Wait" is suppressed.

```python
# Hypothesis: most remaining cut texts are exact loops; characterize their units' first words and onset
def loop_onset(text, p):
    # walk back while text[i]==text[i+p]
    n=len(text); i=n-p-1
    while i>=0 and text[i]==text[i+p]: i-=1
    return i+1
cut=[(x[0],x[1],x[2]) for x in info if x[2]['cut']]
rows=[]
for pid,src,s in cut:
    t=s['text']; p=tail_period(t)
    if p:
        on=loop_onset(t,p); unit=t[-p:]
        # align unit to start at a newline if possible
        j=unit.find('\n')
        u2=(unit[j+1:]+unit[:j+1]) if j>=0 else unit
        rows.append((pid,src,p,on,len(t),u2))
    else:
        rows.append((pid,src,None,None,len(t),''))
nl=sum(1 for r in rows if r[2]); print('cut',len(rows),'exact loops',nl)
print('period median',np.median([r[2] for r in rows if r[2]]),'onset frac median',np.median([r[3]/r[4] for r in rows if r[2]]))
fw=collections.Counter()
for r in rows:
    if r[2]:
        w=re.findall(r"[A-Za-z]+|\d+|\S", r[5].strip()[:30]); fw[w[0] if w else '']+=1
print(fw.most_common(15))
for r in rows[:12]:
    print(r[1][:4], r[0][-10:], 'P',r[2],'on',r[3],'len',r[4],'|',repr(r[5][:150]))
```