```python
# Hypothesis: build a stratified diagnostic set (MATH-500-like level/subject mix + GSM8K) and start notes.
import random
rng = random.Random(1234)
lvl_w = {'Level 1':43,'Level 2':90,'Level 3':105,'Level 4':128,'Level 5':134}
by_lvl = {}
for p in mth:
    by_lvl.setdefault(p['level'], []).append(p)
N_MATH = 384
diag_math = []
for L,w in lvl_w.items():
    k = round(N_MATH*w/500)
    diag_math += rng.sample(by_lvl[L], k)
diag_math = diag_math[:N_MATH]
diag_gsm = rng.sample(gsm, 512-len(diag_math))
diag_ids = [p['id'] for p in diag_math] + [p['id'] for p in diag_gsm]
print(len(diag_math), len(diag_gsm), len(diag_ids), len(set(diag_ids)))
print(pd.Series([p['level'] for p in diag_math]).value_counts().sort_index().to_dict())
print(pd.Series([p['subject'] for p in diag_math]).value_counts().to_dict())
json.dump(diag_ids, open('/work/diag_ids.json','w'))
open('/work/notes.md','w').write("""# Notes
## Setup
- RMSNorm weights are zero-centered (1+w) except linear_attn.norm (plain).
- Diagnostic set: /work/diag_ids.json (384 MATH stratified by MATH-500 level mix + 128 GSM8K), seed 1234.
""")
```