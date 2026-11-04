All generation passes, including the original baseline, used the evaluation's full 32,768-token limit. So the measured gains (E6 about +31 points paired on two held-out sets, cut share 90% to 49%) were obtained under the real protocol.

With 3 passes left, the useful question is whether the remaining failures, about half of samples still cut off, contain a mechanism I can explain and fix without the "migration" that sank E7. The notes say about 30% of E6's cut-offs are "noloop" (no exact verbatim loop). I don't yet know what those are: near-loops with counters, very long genuine reasoning, or degenerate text. I'll re-check the cut-off categories across all E6 runs and read short tails of the noloop ones. I'll also list the helper functions still in the kernel so I can reuse my loop detector.

```python
# Characterize E6 cut-offs (all E6 runs: D gen_16, H gen_18_edited, H2 gen_21_edited): loop vs no-loop, and peek at no-loop tails
fns=[k for k,v in globals().items() if type(v).__name__=='function' and not k.startswith('_')]
print('functions:', fns)
```