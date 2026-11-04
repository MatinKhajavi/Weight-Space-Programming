Each sample records the prediction, a correctness flag, its token count and a cut flag. 67 of the 128 GSM8K answers under Edit A were cut off at 32,768 tokens. Now I'll read what happens after the model first writes the correct answer. For each cut GSM8K answer I find the first occurrence of the reference number and count how often it recurs. I check whether the model ever wrote a \boxed{} or a "final answer" phrase inside its thinking, and I print short excerpts after the first occurrence for a few samples. If the excerpts show the model restating the correct result and then immediately saying "Wait" to re-verify or reinterpret, the failure is at the commit decision. If instead the model drifts into new, wrong computations, the correct number appearing early may be coincidental, and the commit hypothesis weakens.

```python
# Diagnose "failure to commit": what follows the first appearance of the correct answer in edit-A GSM cut samples?
stats = []
for r in gsm_cut:
    t = r['samples'][0]['text']; a = str(r['answer']).strip().replace(',','')
    tt = t.replace(',','')
    occ = [m.start() for m in re.finditer(r'(?<![\d.])'+re.escape(a)+r'(?![\d])', tt)]
    stats.append(dict(id=r['id'], n=len(tt), nocc=len(occ), first=(occ[0]/len(tt) if occ else None),
                      boxed=tt.count('\\boxed'), fa=len(re.findall(r'(?i)final answer', tt)), think_end='</think>' in t))
df_s = pd.DataFrame(stats)
print('has ans:', (df_s.nocc>0).mean().round(2), ' median first frac', df_s.first.median(), ' median nocc', df_s.nocc.median())
print('with boxed in text:', (df_s.boxed>0).mean().round(2), ' with "final answer":', (df_s.fa>0).mean().round(2))
for r in gsm_cut[:4]:
    t = r['samples'][0]['text'].replace(',',''); a = str(r['answer']).strip().replace(',','')
    occ = [m.start() for m in re.finditer(r'(?<![\d.])'+re.escape(a)+r'(?![\d])', t)]
    if not occ: continue
    i0 = occ[0]
    print('==', r['id'], 'ans', a, 'len', len(t), 'first at', i0)
    print(repr(t[max(0,i0-120):i0+260]))
```