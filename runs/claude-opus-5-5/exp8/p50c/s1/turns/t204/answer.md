Outputs are stored as float16, so this request would produce about 4.7 GB, under the 6 GB limit. However, I noticed a practical problem the notes don't resolve. A teacher-forced request must be pasted inline as JSON in my reply, and this one is about 1.2 million characters, which cannot be pasted. The earlier "corpus1" teacher-forced pass in the notes also appears never to have run, probably for the same reason. Also, the 16th pass (gen_21) has no notes entry. I will check what gen_21 was, and how large the earlier request files were, before planning further passes.

```python
# Recover undocumented pass gen_21 and sizes of earlier TF request files (to know what TF payload sizes are feasible)
for f in ['/work/req_tfC.json', '/work/req_tf_conc.json', '/work/tf_wins_A.json', '/work/tf_wins_C.json', '/work/tf_conc_texts.json', '/work/tf_corpus1_texts.json']:
    print(f, os.path.getsize(f))
rows = [json.loads(l) for l in open('/work/gen_21.jsonl')]
print(len(rows), list(rows[0].keys()))
from collections import Counter
print(Counter(r.get('source') for r in rows), Counter(r.get('model', '?') for r in rows))
print('acc', np.mean([bool(r.get('correct')) for r in rows]), 'n_tokens mean', np.mean([r.get('n_tokens', 0) for r in rows]))
print(Counter(r.get('finish_reason', r.get('status', '?')) for r in rows))
print([ (r['id'], r.get('correct'), r.get('n_tokens')) for r in rows[:5]])
```