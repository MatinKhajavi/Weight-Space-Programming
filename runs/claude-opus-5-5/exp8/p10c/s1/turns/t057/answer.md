The failure shows that teacher-forced texts have to be pasted literally into the forward block. Since I must not print whole generated transcripts, I cannot feed the model's own long generations back in. Dense per-token data is therefore limited to the two loop texts I already have (all layers, all MLP activations, attention) and to short texts I write myself. Generation passes, which accept problem ids, remain the scalable tool for causal tests. So my main evidence will be weight analysis plus the per-token data in fwd_4, followed by paired generation tests of edits. Before choosing between damping the copying heads and weakening the drive to write "Wait", I want to see how successful runs leave the thinking. Specifically: does a concluding phrase usually precede `</think>`, so that the exit depends on a "doubt again vs. conclude" choice at a line start? If finishers typically end with a concluding line, weakening the "Wait" choice at line starts would directly open that exit. If they end abruptly, the exit is a different decision. I print only short endings (≤250 characters) of a few finished samples and count how the last lines before `</think>` start.

```python
# Hypothesis: finishers exit via a concluding line chosen at a line start (instead of another 'Wait' bullet)
fin_s = [(i, m) for i, m in enumerate(meta_s) if not m[3]]
ends = []
for i, m in fin_s:
    t = texts[i]; k = t.find('</think>')
    if k < 0: continue
    ends.append((i, t[:k]))
for i, pre in ends[:4]:
    print(meta_s[i][:3], repr(safe(pre[-220:])))
# first word of the last 3 lines before </think>, across all finishers
from collections import Counter
c = Counter()
for i, pre in ends:
    lines = [l.strip(' *-') for l in pre.strip().split('\n') if l.strip()]
    for l in lines[-3:]: c[(l.split() or [''])[0][:12]] += 1
print(len(ends), c.most_common(15))
```