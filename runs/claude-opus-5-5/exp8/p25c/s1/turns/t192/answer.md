The specificity check is clean. Only 25 of the roughly 1,700 prompt and lead tokens change by more than one nat under E6, and 21 of them are the targeted openers ("Wait" 15, "Okay" 2, "Let" 2, "This" 1, "So" 1). Math content, numbers and formatting are essentially untouched.

At line-start positions inside loops, the original's probability of the repeated continuation climbs from 0.90 to 0.996 with repetition. Under E6 it stays flat around 0.80–0.85. The probability mass on "Wait" at those positions falls to about 0 by the fourth repetition (original: 0.18). That line-start set mixes indentation/bullet positions with opener positions, so I'll restrict it to the exact positions where the original text has "Wait" or another targeted opener. These are the decision points the edit is meant to change.

I'll also compute the layer-by-layer logit attribution of the opener margin at those points, with the same method as my earlier notes. The question is which components now push against the repeated opener, and what compensates (previously: the layer-22 MLP relaxing its suppression).

```python
# Refined decision points: P(actual opener) at positions where the text has a set opener, by repetition k (orig vs E6); 'Wait' only too
Rw=[];
for b in range(14):
    T=L4[b]; rk=rep_index(b); ids=ids4[b,:T]
    for t in range(T-1):
        y=ids[t+1]
        if y in opener_ids: Rw.append((b,t,rk[t+1], y in wait_ids, np.exp(LPo[b,t]), np.exp(LP6[b,t]), np.exp(LP['E3'][b,t]) if 'E3' in LP else np.nan))
Rw=pd.DataFrame(Rw, columns=['b','t','k','wait','p_orig','p_E6','p_E3'])
print('opener positions by k (all set openers):'); print(Rw.groupby('k')[['p_orig','p_E3','p_E6']].mean().round(3).T, Rw.groupby('k').size().tolist())
print('Wait-only:'); print(Rw[Rw.wait].groupby('k')[['p_orig','p_E3','p_E6']].mean().round(3).T)
# what does E6 put in place of Wait at k>=3 Wait positions? top-3 alternatives (ids -> short ascii)
alt=Counter()
for _,r in Rw[(Rw.wait)&(Rw.k>=3)].iterrows():
    for j in range(3): alt[asc(tok.decode([int(TKI6[int(r.b),int(r.t),j])])).strip()[:8]]+= 1
print('E6 top-3 at Wait positions k>=3:', alt.most_common(12))
```