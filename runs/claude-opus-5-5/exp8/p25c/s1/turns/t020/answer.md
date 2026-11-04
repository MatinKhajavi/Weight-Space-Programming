The arrays are float16 with shapes (512 samples, 14 residual layers, 1024) and (512, 13 MLP layers, 3584), one row per sample in the index order, with each sample's grade and token count. Since the two samples of a problem share the same prompt, the last-prompt-token activations cannot explain why one sample loops and the other does not, so I will focus on the answer-averaged activations. I will compute, per layer, the difference between finished and looping samples and measure how well a single direction separates them, both overall and within problems that had one finished and one looping sample (which controls for problem difficulty). A strong within-problem separation concentrated at specific layers or MLP neurons would suggest candidate components to test causally; weak or diffuse separation would mean the averaged activations are dominated by content and I should rely on token-level teacher-forced analysis instead.

```python
# Contrast finished vs looping samples in answer-averaged activations, per layer
samp = idx['samples']
fin_mask = np.array([s['correct'] for s in samp])  # all finished ones were correct
items = [s['item'] for s in samp]
from collections import defaultdict
byitem = defaultdict(list)
for i,s in enumerate(samp): byitem[s['item']].append(i)
mixed = [v for v in byitem.values() if len(v)==2 and fin_mask[v[0]]!=fin_mask[v[1]]]
print('finished', fin_mask.sum(), 'mixed problems', len(mixed))
from sklearn.metrics import roc_auc_score
Hm = np.asarray(H_mean, dtype=np.float32)
for li, L in enumerate(idx['layers']):
    X = Hm[:, li]
    d = X[fin_mask].mean(0) - X[~fin_mask].mean(0)
    proj = X @ d / np.linalg.norm(d)
    auc = roc_auc_score(fin_mask, proj)
    # within mixed pairs: fraction where finished sample projects higher (direction from all data, slight leakage)
    w = np.mean([ (proj[a] > proj[b]) if fin_mask[a] else (proj[b] > proj[a]) for a,b in mixed])
    print(L, 'AUC %.3f'%auc, 'pairwise %.2f'%w, 'dnorm %.2f'%np.linalg.norm(d), 'xnorm %.1f'%np.linalg.norm(X,axis=1).mean())
```