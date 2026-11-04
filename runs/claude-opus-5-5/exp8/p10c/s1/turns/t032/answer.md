```python
# Hypothesis: build two natural loop texts (<=2040 tokens) for the attention deep-dive; print JSON-escaped for pasting.
deep = []
for i in [417, 437]:
    s_ = samples[i]; ids = toks(s_['text']); pt = prompt_text(s_['id'])
    npr = len(toks(pt))
    txt = pt + tok.decode(ids[:2040 - npr])
    deep.append(txt)
    print(s_['id'], len(toks(txt)), len(json.dumps(txt)))
json.dump(deep, open('/work/deep_texts.json','w'))
djs = json.dumps(deep)
print(len(djs))
print(djs[:9000])
```