```python
# Hypothesis: shorten texts to 1950 tokens to leave margin for tool-side tokenization differences; reprint.
deep = []
for i in [417, 437]:
    s_ = samples[i]; ids = toks(s_['text']); pt = prompt_text(s_['id'])
    npr = len(toks(pt))
    txt = pt + tok.decode(ids[:1950 - npr])
    deep.append(txt)
    print(s_['id'], len(toks(txt)), len(json.dumps(txt)), txt.count('Wait, if the cat is stationary'), txt.count('If he has 8 available, he has *used* 2 bones?'))
json.dump(deep, open('/work/deep_texts.json','w'))
djs = json.dumps(deep)
print(len(djs))
# print only the tails (the parts that differ from before) to minimize re-copy errors
print(repr(deep[0][-700:]))
print(repr(deep[1][-500:]))
```