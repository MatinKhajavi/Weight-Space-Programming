```python
# Hypothesis: see how a GSM8K transcript proceeds after solving (what keeps it from closing </think>).
r = [r for r in rows if r['source']=='gsm8k'][2]
t = r['samples'][0]['text']
print(r['id'], 'answer', r['answer'], 'cut', r['samples'][0]['cut'])
print(asc(ID[r['id']]['problem'][:300]))
print('----')
print(asc(t[:3500]))
```