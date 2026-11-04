The answer-string check is too crude: short answers like "2" appear in almost any text, and my old `fracs` array has 449 entries, so it doesn't line up with the 470 cut samples. What is clear is that 368 of 470 loop units contain digits. Most loops repeat a computation or verification line ("Wait, $88 \div 4 \div 2 = 11$."), not a final-answer wrap-up. I'll recompute loop onset and period for every cut sample with a clean exact-periodicity detector, convert onsets to token positions, and print a few short excerpts around the onset to see how loops begin. This tells me whether loops are entered abruptly (one line repeated immediately) or after a gradual narrowing.

```python
# Hypothesis: loops are exact char-periodic tails; measure onset (tokens), period, and look at how they start
def find_loop(t):
    tail=t[-6000:]
    for P in range(4,1800):
        if tail[-P:]==tail[-2*P:-P] and tail[-2*P:-P]==tail[-3*P:-2*P]:
            i=len(t)-P-1
            while i>=0 and t[i]==t[i+P]: i-=1
            return i+1,P
    return None,None
LP=[]
for r,s in cut_s:
    on,P=find_loop(s['text']); 
    ntok_on=len(tok.encode(s['text'][:on]).ids) if on is not None else None
    LP.append((r['id'],on,P,ntok_on))
ok_=[x for x in LP if x[1] is not None]
print('loops found',len(ok_),'of',len(LP))
print('onset tokens pct',np.percentile([x[3] for x in ok_],[10,25,50,75,90]).astype(int))
print('period chars pct',np.percentile([x[2] for x in ok_],[10,25,50,75,90]).astype(int))
for (r,s),x in list(zip(cut_s,LP))[5:9]:
    if x[1]: print('==',x[0],'P',x[2],'onTok',x[3],'|',repr(s['text'][x[1]-250:x[1]+x[2]+60]).encode('ascii','replace').decode()[:420])
```