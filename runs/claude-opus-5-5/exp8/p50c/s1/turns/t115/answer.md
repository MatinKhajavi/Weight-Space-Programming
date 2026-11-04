Without Edit A's head ablation, the brake alone reaches 44.9% (GSM8K 56.3%, MATH 33.6%) with 53.9% still cut off. The full factorial on S1 is now: original 12.5%, head ablation only 35.2%, brake only 44.9%, both 60.2%. The two repairs address different mechanisms and both are needed. The gains are roughly additive (+22.7 and +32.4 alone, +47.7 together, slightly sub-additive near the ceiling). This matches the diagnosis: verbatim copying by induction heads produces exact loops, and the self-reinforcing "Wait" spiral produces endless doubting. I'll check that the brake-only run's cut-offs are mostly the original's exact loops; that would confirm the brake does not remove copy loops. Then I'll look closely at the remaining loops under the combined edit, the best candidate (74 cut, 47 exact loops), to find what drives them now that "Wait" is gone. I'll print the period and a short excerpt of the repeating unit for several of them.

```python
# Factorial check + characterize remaining loops under A+brake(x2)
gB = readgen('/work/gen_15.jsonl')
pB = [tail_period(gB[i]['text']) for i in S1 if gB[i]['cut']]
print('brake-only: cut %d, exact loops %d; Wait/1k median %.2f; vs orig %+.1f SE %.1f (W->R %d R->W %d); A+brake vs brake %+.1f SE %.1f (%d/%d)' % (
    sum(gB[i]['cut'] for i in S1), sum(p is not None for p in pB), np.median([1000*len(re.findall(r'\bWait\b', gB[i]['text']))/gB[i]['ntok'] for i in S1]),
    *paired(gB, gO, S1), *paired(gC, gB, S1)))
cnt = 0
for i, p_ in zip(cutC, pers):
    if p_ and cnt < 7:
        t = gC[i]['text']; unit = t[-p_:]
        # onset: scan back while periodic
        k = len(t) - p_
        while k-1 >= p_ and t[k-1] == t[k-1+p_]: k -= 1
        print('==', i, 'period %d chars, loop onset at char %d of %d (%.0f%%)' % (p_, k, len(t), 100*k/len(t)))
        print('   unit:', repr(re.sub(r'\s+', ' ', unit)[:160].encode('ascii','replace').decode()))
        cnt += 1
```