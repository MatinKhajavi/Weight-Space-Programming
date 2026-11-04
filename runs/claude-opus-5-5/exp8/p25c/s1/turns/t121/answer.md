E6 (eight openers) reached 43.2% on the D set (GSM8K 55.7%, MATH 35.6%), against 42.2% for E5. Every item's sampling seed is fixed by the problem, so I can pair gen_16 with gen_14 (E5) and gen_12 (E4b) myself, sample by sample. I expect the E6-vs-E5 gain to be small, about one point and within its standard error, which would mean the opener-subspace approach is saturating. First I will list the helper functions and variables still in the kernel and the record format, so I can reuse earlier loop-analysis code. Then I will compute the paired E6-vs-E5 difference.

```python
# Hypothesis: E6 vs E5 paired gain on D is small (~+1 pt, within SE); first recover kernel helpers + record schema.
import json, collections, numpy as np, types
fns = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, types.FunctionType)]
print("functions:", fns)
others = [k for k,v in globals().items() if not k.startswith('_') and not isinstance(v,(types.FunctionType, types.ModuleType, type))]
print("vars:", others[:150])
def readj(p): return [json.loads(l) for l in open(p)]
g16 = readj('/work/gen_16.jsonl'); g14 = readj('/work/gen_14.jsonl')
print(list(g16[0].keys()))
print({k:(str(v)[:80]) for k,v in g16[0].items() if k not in ('text','output','completion')})
```