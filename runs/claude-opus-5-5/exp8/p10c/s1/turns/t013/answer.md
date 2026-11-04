```python
# Hypothesis: characterize cut generations - does </think> ever appear, what does the tail look like (loops/drift)?
samples = []
for g in gen:
    for s_ in g['samples']:
        samples.append(dict(id=g['id'], source=g['source'], ref=g['answer'], **{k:v for k,v in s_.items()}))
print(samples[0].keys())
df = pd.DataFrame([{k:v for k,v in s_.items() if k!='text'} for s_ in samples])
print(df.head(3).to_string()[:1500])
df['has_close'] = [('</think>' in s_['text']) for s_ in samples]
df['len_chars'] = [len(s_['text']) for s_ in samples]
print(df['has_close'].mean())
```