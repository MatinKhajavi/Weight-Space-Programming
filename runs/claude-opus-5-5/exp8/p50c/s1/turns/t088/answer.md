The gen_6 file stores each problem as an id, a source, the reference answer and a list of samples. I still need the fields of one sample, specifically how a cut-off answer is flagged and whether a token count is stored. Next I'll pick out the GSM8K samples from the Edit-A run that were cut off at the limit. In each I'll locate the first appearance of the correct answer, then read what follows. This step only sets that up: it prints the sample fields and counts the cut GSM8K samples. If the cut flag sits in a field I don't expect, the count will come out as zero and I'll adjust before reading any text.

```python
# Inspect sample structure and pick GSM cut samples where the correct answer appears before the loop
s0 = g6e[0]['samples'][0]; print({k:(str(v)[:100]) for k,v in s0.items() if k!='text'})
def first_ans_idx(text, ans):
    a = str(ans).strip().replace(',','')
    m = re.search(r'(?<![\d.])'+re.escape(a)+r'(?![\d])', text.replace(',',''))
    return m.start() if m else -1
gsm_cut = [r for r in g6e if r['source']=='gsm8k' and r['samples'][0].get('finish_reason', r['samples'][0].get('cut'))]
print(len(gsm_cut))
```