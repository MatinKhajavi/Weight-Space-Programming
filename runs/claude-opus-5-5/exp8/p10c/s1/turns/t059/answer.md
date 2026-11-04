The per-token data show two phases. Before any loop, P(" Wait") at line starts climbs steadily within a text: from about 0.01–0.02 in the first quarter to 0.07–0.17 in the last quarter. Inside the loop the line-start choices become near-deterministic copies of the previous cycle. This fits in-context self-reinforcement: each earlier "Wait" makes the next one more likely, and the copying heads seen attending to earlier Waits are a plausible carrier. The exact loop is then the extreme case of the same copying. Before committing passes to an edit, I want to rule out something simpler. GSM8K accuracy of 14% with 92.6% cut-offs is far below what a 0.8B reasoning model normally does, so I check the weights for an anomalous component, such as a head or neuron group with abnormal scale. If one tensor slice is a clear outlier against its peers in other layers, that is a strong candidate for what drives the looping. If all statistics are smooth across layers, the looping is a learned behaviour, and I will proceed with the copying-head and doubt analyses.

```python
# Hypothesis check: is there an anomalous component (outlier head/neuron/norm scale) in the weights?
import torch
names = [n for n in NAMES if 'language_model.layers.' in n]
print(len(names)); print(sorted(set(n.split('layers.')[1].split('.',1)[1] for n in names))[:40])
```