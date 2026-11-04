```python
# Hypothesis: practice ids are sequential per subject; check so diagnostic sets can be typed compactly.
from collections import defaultdict
subj_ids = defaultdict(list)
for m in mth: subj_ids[m['subject']].append(m['id'])
for s,v in subj_ids.items(): print(s, len(v), v[:3], v[-1])
print(gsm[:3][0]['id'], gsm[-1]['id'])
# level distribution in first 60 of each subject
for s,v in subj_ids.items():
    print(s, Counter(ID[i]['level'] for i in v[:60]).most_common())
```