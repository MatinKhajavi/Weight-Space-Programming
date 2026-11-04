The randomized within-problem test gives a clear negative result, and it is important. Within the same problem, "my thought process" samples are no more accurate than "a thinking process" samples: +0.5 ± 2.8 points accuracy, +1.5 ± 2.5 points cut rate, and essentially no difference in length (+0.06 ± 0.07 in log tokens). The large pooled gap (85% vs 70% here, 94% vs 59% before) is entirely problem selection, because the model picks "my thought process" for problems it finds easy. The seeds are perfectly reproducible (72 of 72 openings matched).

So the opening, and the nesting style that goes with it, is a difficulty marker, not a cause. I will not edit it, and the opening shift caused by my head ablation is harmless.

I return to the mechanism that produced the biggest win, a native brake neuron. Under the earlier edit A, direct logit attribution at loop-cycle starts found neuron 22.573 opposing the loop token. Under C the loops start with other tokens (So/This/Maybe or mid-computation), and there may be other native brake neurons, ones that push against the loop continuation, which I could strengthen the same way. The existing teacher-forced data on C loop windows (fwd_18) should contain the needed MLP activations. First I check what it holds and how the windows were defined.

```python
# Recall data for a neuron-level DLA at C's loop-cycle starts: fwd_18 contents and window metadata
with open('/work/notes.md','a') as f:
    f.write("- PASS 19 = gen_24 (C, 36 discordant problems x14, 32768): pooled acc my .85 vs a .70, BUT within-problem diff my-a = +0.5 pts SE 2.8 (cut +1.5 SE 2.5, log-tok +0.06 SE .07); seeds reproduce openings 72/72. ==> opening mode is a DIFFICULTY MARKER, not causal. Do not edit it.\n")
m18 = json.load(open('/work/fwd_18/meta.json'))
print(m18['lengths'], m18['edited_tensors'])
print(sorted(m18['files'].keys())[:60])
wC = json.load(open('/work/tf_wins_C.json')); print(type(wC).__name__, (list(wC.keys()) if isinstance(wC,dict) else [ {k:(v if not isinstance(v,(str,list)) else (type(v).__name__, len(v))) for k,v in w.items()} for w in wC]))
```