```python
# Inspect sample structure and pick GSM cut samples where the correct answer appears before the loop
s0 = g6e[0]['samples'][0]; print({k:(str(v)[:100]) for k,v in s0.items() if k!='text'})
def first_ans_idx(text, ans):
    a = str(ans).strip().replace(',','')
    pats = [r'(?<![\d.])'+re.escape(a)+r'(?![\d])']
    m = re.search(pats[0], text.replace(',',''))
    return m.start() if m else -1
gsm_cut = [r for r in g6e if r['source']=='gsm8k' and r['samples'][0].get('finish_reason', r['samples'][0].get('cut'))]
print(len(gsm_cut))
```