```python
# Hypothesis: inspect practice data composition (levels/subjects) to design a representative diagnostic set.
import pandas as pd
def readj(p): return [json.loads(l) for l in open(p)]
gsm = readj('/practice/gsm8k_train.jsonl'); mth = readj('/practice/math_train.jsonl'); aime = readj('/practice/aime.jsonl')
print(gsm[0].keys(), mth[0].keys(), aime[0].keys())
dm = pd.DataFrame(mth)
print(dm.groupby(['level']).size())
print(dm.groupby(['subject']).size())
print(gsm[0]['id'], gsm[0]['problem'][:200], '|', gsm[0]['answer'])
print(mth[0]['id'], mth[0]['problem'][:200], '|', mth[0]['answer'])
print(aime[0]['id'], aime[0]['answer'])
```