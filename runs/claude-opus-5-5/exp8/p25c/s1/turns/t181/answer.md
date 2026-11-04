Layer 23 head 2 does not support the anti-repetition idea; I'm dropping the boost.
- **Head 2:** it attends almost entirely to the current position (96% in the prompt, 90% in the lead). With its prediction-suppressing value/output map, that dampens whatever the model is about to predict. It is a general calibration mechanism. Its attention to earlier same tokens does rise in loops (0.07 → 0.29), but strengthening it would mainly flatten the whole output distribution, a global knob this task excludes.
- **Head 6:** it shifts from self-attention (0.95 in the prompt) to earlier same-token positions (0.37 in the lead, 0.88 by the sixth repetition).
- **Head 5:** its induction share grows (0.06 → 0.18).

Together with E8's collapse, this shows these two heads carry copying that coherent reasoning depends on. The loop attractor reuses that machinery, so it cannot be cut out at the head level.

The opener-level lever is saturated (E7 migration, dose plateau), so the main remaining question is where E6's leftover loops start. My notes say about half of easy-problem cut-offs already contain the right answer, so many leftover loops may begin after the problem is solved. I'll extract, for 12 E6 exact loops, a short excerpt just before onset and the loop period itself. A consistent trigger, such as a particular self-check pattern right after the answer, would be a candidate for one more targeted test. Heterogeneous triggers would mean finalizing E6.

```python
# What precedes E6 loop onsets? Short excerpt before onset + first period, for 12 exact loops (D/H/H2), plus onset position stats
rng_=np.random.default_rng(7); sel=rng_.choice(len(lo), 12, replace=False)
ons=np.array([x[3][1]/max(1,len(x[2]['text'])) for x in lo]); print('onset as frac of text: q', np.percentile(ons,[10,25,50,75,90]).round(2))
on_tok=[]
for j in sel:
    k,i,s,(p,o)=lo[j]; t=s['text']
    pre=asc(t[max(0,o-170):o]).replace('\n','|'); per_=asc(t[o:o+min(p,150)]).replace('\n','|')
    print(f'== {k} {i[-26:]} P={p} on={o/len(t):.2f} | PRE: ...{pre} || LOOP: {per_}')
```