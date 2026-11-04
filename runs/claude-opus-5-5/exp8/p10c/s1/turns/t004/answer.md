```python
# Hypothesis: inspect practice data format and distribution (levels/subjects) to design diagnostic sampling.
import pandas as pd
def readjl(p): return [json.loads(l) for l in open(p)]
gsm = readjl('/practice/gsm8k_train.jsonl'); mth = readjl('/practice/math_train.jsonl'); aime = readjl('/practice/aime.jsonl')
print(gsm[0].keys(), mth[0].keys(), aime[0].keys())
print(gsm[0]['id'], gsm[0]['problem'][:200], '| ans:', gsm[0]['answer'])
print(mth[0]['id'], mth[0]['subject'], mth[0]['level'], mth[0]['problem'][:200], '| ans:', mth[0]['answer'])
dm = pd.DataFrame(mth)
print(dm.groupby(['subject','level']).size().unstack())
```