Both windows contain clean exact loops, with token periods of 85 and 64 starting around token 720–760, giving 52 line-start branch points inside the loops. One detail matters for the attribution: the final-norm weight has mean 3.3, so I must use the same convention (w or 1+w) as my earlier attribution helpers. I will read the convention from the existing helper's source, then compute, for each neuron in layers 14–23, the average contribution at the branch points to the logit of the copied token minus the best alternative. As a control, I compute the same at line starts outside the loop region of the same windows. A real brake should be clearly negative at loop branch points, and preferably more strongly engaged there than at ordinary line starts.

```python
# Hypothesis: a few late-MLP neurons consistently push AGAINST the copied token at loop branch points (native brakes)
import inspect
try:
    src = inspect.getsource(neuron_dla); print('helper uses 1+w:', ('1+' in src.replace(' ','')) , '|', A(src[:300]).replace('\n',' | '))
except Exception as e: print('no source', e)
```