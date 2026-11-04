This example changes the picture. For "Compute sin 0°" the model has the answer within the first few hundred characters. By about 15,000 characters it writes "Okay, ready to generate the response. I will write it out now." But it never emits </think>. It keeps re-planning ("Wait, I should check...") and drifts into degenerate lists ("I will use `\mathbf`...") until an exact loop locks in. So the exact loop may be the final stage of a longer failure to stop after the problem is solved. If that holds across many texts, then breaking exact loops alone may just turn loops into rambling, and both stages matter. To test this, I will measure over all 443 looping answers how often the correct answer already appears in the text before the loop starts. I will also count "wrap-up" statements ("final answer", "ready to", "I will write", \boxed) before the loop.

```python
# Hypothesis: many loops happen AFTER the model has already found the correct answer (failure to stop, then degeneration)
import re
def norm_ans(a): return re.sub(r'\s+','',str(a)).replace('\\!','').replace('dfrac','frac').replace('tfrac','frac')
def ans_in(text, ans, src):
    a = norm_ans(ans); tt = re.sub(r'\s+','',text).replace('dfrac','frac').replace('tfrac','frac')
    if src=='gsm':
        a2 = a.replace(',','')
        return re.search(r'(?<![\d.])'+re.escape(a2)+r'(?![\d])', tt.replace(',','')) is not None
    return (('boxed{'+a+'}') in tt) or (('='+a) in tt and len(a)>=1)
phr_list = ['final answer','ready to','I will write','Final Answer','boxed{','</think>','Let me write','write the final','write the response']
rows_=[]
for _, r in c2.iterrows():
    pre = r['text'][:int(r['loopstart'])]
    ref = pid2prob[r['id']]['answer']
    rows_.append(dict(src=r['src'], ans_pre=ans_in(pre, ref, r['src']), onset=r['onset_tok'],
                      **{p_: pre.count(p_) for p_ in phr_list}))
Q = pd.DataFrame(rows_)
print('share with correct answer already present before loop:', Q.groupby('src').ans_pre.mean().round(2).to_dict(), 'overall', round(Q.ans_pre.mean(),2))
print('share with phrase before loop:', {p_: round((Q[p_]>0).mean(),2) for p_ in phr_list})
print('median count when present:', {p_: int(Q.loc[Q[p_]>0, p_].median()) if (Q[p_]>0).any() else 0 for p_ in phr_list})
print('think-close token id:', tok.token_to_id('</think>'), 'boxed ids', tok.encode('\\boxed{').ids)
```