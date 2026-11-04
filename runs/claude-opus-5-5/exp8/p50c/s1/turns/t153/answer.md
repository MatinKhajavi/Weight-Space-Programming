Across all loop tokens, the remaining non-ablated induction heads are 11.1 (0.66-0.69, but 0.61 before the loop as well, so general), 23.4 (0.52/0.54 vs 0.36 pre-loop), 23.7 (0.44 vs 0.27) and 19.5 (0.41-0.49 vs 0.33). Apart from 11.1, these are exactly the Edit B heads whose joint removal hurt accuracy. The remaining copy drive is therefore spread over heads that also do legitimate work, plus the L21 DeltaNet recall, and further ablation looks unpromising. The other lever is the model's own anti-repetition machinery, as with the Wait brake. The weight analysis found strong copy-suppression heads: 23.2 (score -7.6), 11.0 (-7.0), 11.6 (-11.4) and 15.3 (-3.0). I'll check whether they attend to the about-to-be-repeated tokens inside loops. If one does, amplifying it would counter the copy without touching the heads needed for computation.

```python
# Hypothesis: copy-suppression heads (23.2, 11.0, 11.6, 15.3) attend to the about-to-be-copied tokens in loops
for (l,h) in [(23,2),(11,0),(11,6),(15,3),(15,0),(19,4)]:
    print(f'{l}.{h}', 'W2-loop',round(r_w2[(l,h)],2),'W1-loop',round(r_w1in[(l,h)],2),'W1-pre',round(r_w1pre[(l,h)],2))
# also attention to same-token earlier positions (duplicate-token attention) at decision points in W1 loop lines
def dup_mass(b, seq, l, h, qs):
    ms=[]
    for q in qs:
        tg=[j for j in range(q) if seq[j]==seq[q+1]]   # earlier occurrences of the NEXT token (what is about to be predicted)
        if tg: ms.append(float(Att[l][b,h,q,tg].sum()))
    return np.mean(ms) if ms else float('nan')
qs_in=[p-1 for p in fcW1 if p>=690]; qs_pre=[p-1 for p in fcW1 if p<673]
for (l,h) in [(23,2),(11,0),(11,6),(15,3),(23,4),(23,7),(19,5)]:
    print(f'{l}.{h} attn on earlier copies of next token: loop',round(dup_mass(0,seq0,l,h,qs_in),2),'pre',round(dup_mass(0,seq0,l,h,qs_pre),2))
```