fwd_4 was a teacher-forced pass on two of the model's own texts that loop early (875 and 1,544 tokens), with hidden states at every layer and full-attention patterns. Loops typically start at about 20% of the 32k budget (roughly 6.5k tokens) and then run to the limit. The run-36 table compared a per-layer readout between pre-loop and in-loop positions of probe 2. The in-loop excess appeared first after full-attention layer 15 (+2.5), then after layers 19 and 23. That points to attention-based copying from the previous cycle (induction-like heads) as the loop driver. `res` appears to hold per-head attention statistics, so I'll print it to see which heads concentrate their attention on the previous loop cycle. I'll also confirm what `a_pre`/`a_loop` measured.

```python
# Hypothesis: res holds per-head loop-attention stats (layer, head, ...); find heads that attend to the previous cycle in loops
print(len(res)); 
for t in sorted(res, key=lambda t: -float(t[-1]))[:16]: print(tuple(round(float(x),3) if i>1 else x for i,x in enumerate(t)))
print('a_pre', a_pre.round(2)[-6:], 'a_loop', a_loop.round(2)[-6:])
print(open('/work/notes.md').read()[:200])
print('per',per,'loop1',loop1[:10],'ind',ind[:10], 'top', top)
```