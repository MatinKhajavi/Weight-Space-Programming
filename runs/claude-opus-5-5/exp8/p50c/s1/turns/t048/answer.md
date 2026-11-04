Several full-attention heads become strongly locked onto the previous loop cycle. Head 15.7 puts 0.89 of its attention on induction targets inside the loop versus 0.47 on pre-loop text, and head 11.1 gives 0.76 versus 0.49. Layer 23 (heads 5, 0, 7, 4) and layer 19 (heads 5, 7) roughly double their induction mass inside loops. This fits the readout jump after layer 15. Before choosing a mechanism to edit, I want to understand what the loops are about. If most loops happen after the model already has the answer and is trying to wrap up ("Wait, I'll write \boxed{2}"), then breaking the loop could convert many cut-offs into correct answers. I'll check how often the reference answer appears before the loop starts, and classify loop units by content (boxed/answer/final versus mid-computation).

```python
# Hypothesis: many loops start after the model already found the answer (wrap-up loops), so breaking them would pay off
import re
def norm_ans(a): return re.sub(r'[\s\$]','',str(a)).replace('\\!','').replace('dfrac','frac')
cut_s=[(r,s) for r in rows for s in r['samples'] if s['cut']]
print('cut with pred:', sum(s['pred'] not in (None,'None') for r,s in cut_s), 'cut&correct', sum(bool(s['correct']) for r,s in cut_s))
has_box_ans=0; has_ans_pre=0; kinds=collections.Counter()
for i,(r,s) in enumerate(cut_s):
    t=s['text']; on=int(fracs[i]*len(t)) if i<len(fracs) else len(t)//5
    pre=t[:on]; a=norm_ans(r['answer'])
    boxes=[norm_ans(b) for b in re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', pre)]
    if a in boxes: has_box_ans+=1
    if a and a in re.sub(r'[\s\$]','',pre): has_ans_pre+=1
    u=units[i] if i<len(units) else ''
    k='boxed' if 'boxed' in u else ('answer/final' if re.search(r'answer|final|result',u,re.I) else ('digits' if re.search(r'\d',u) else 'other'))
    kinds[k]+=1
print('n',len(cut_s),'boxed correct ans before loop',has_box_ans,'answer string before loop',has_ans_pre); print(kinds)
```