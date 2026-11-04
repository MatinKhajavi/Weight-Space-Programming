```python
# Hypothesis: on GSM8K the model finds the answer early but never closes thinking; check when the correct answer first appears.
def first_answer_pos(t, ans):
    # first position where answer string appears as a standalone number
    m = re.search(r'(?<![\d.])'+re.escape(ans)+r'(?![\d])', t)
    return m.start() if m else None
g = dfi[dfi.src=='gsm8k']
pos=[]
for r in rows:
    if r['source']!='gsm8k': continue
    t=r['samples'][0]['text']; a=r['answer'].strip()
    fp = first_answer_pos(t,a)
    pos.append((fp, len(tok.encode(t[:fp]).ids) if fp is not None else None, r['samples'][0]['cut']))
pp=pd.DataFrame(pos, columns=['char','tok','cut'])
print('answer string appears:', pp.char.notna().mean())
print(pp.tok.describe())
```