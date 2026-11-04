The finished-but-wrong answers are mostly genuine reasoning errors, not grading or format mismatches. GSM8K errors are typically factor-of-two slips or misread relationships (90 vs 180, 30 vs 60, 72 vs 36, 6 vs 12). Only a handful involve text or units inside the box, and even those are numerically wrong. One pattern stands out: 8 MATH answers are "0" (plus a few "1"), often from texts of 20k–33k tokens. That suggests a fallback guess after very long, unresolved reasoning rather than a computed result. Reasoning errors are a capability limit that a targeted edit is unlikely to repair, so the cut-off answers (about 25%) remain the best lever. To find a mechanism there, I will read short excerpts at 30%, 60% and 90% of several cut texts under C, half GSM8K and half MATH. I also want to see what the long non-looping runs are doing and how the "0" fallback answers end.

```python
# Qualitative look: what C cut texts are doing at 30/60/90% of their length, and how the '0' fallback answers end
cutS=[(r_,s_) for (_,r_,s_) in allrecs if s_['cut']]
pick=[x for x in cutS if 'gsm' in x[0]['id']][:4]+[x for x in cutS if 'math' in x[0]['id']][:4]
for r_,s_ in pick:
    t_=s_['text']; P_,on_=tail_period(t_)
    print(f"== {r_['id']} ans {r_['answer'][:20]} len {len(t_)} loopP {P_} onset {on_}")
    for fr in (0.3,0.6,0.9):
        k_=int(fr*len(t_)); print(f'  [{fr}]', repr(asc(t_[k_:k_+230])))
zero_cases=[(r_,s_) for (_,r_,s_) in allrecs if (not s_['correct']) and (not s_['cut']) and str(s_['pred'])=='0'][:3]
for r_,s_ in zero_cases:
    t_=s_['text']; k_=t_.find('</think>'); print('ZERO', r_['id'], 'ans', r_['answer'][:15], '| before </think>:', repr(asc(t_[max(0,k_-250):k_]))[:270])
```