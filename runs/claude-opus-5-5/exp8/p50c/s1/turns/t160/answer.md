A key detail: in the W1 loop the runner-up to repeating the line is " Wait" in 22 of 29 lines. Inside a copy loop, "Wait" is the model's natural exit, and the brake suppresses it. The lock-in itself is spread out: from the first to the late repeats the copy margin grows from 1.2 to 5.5 logits. That growth comes from the L20 (+2.6), L19 (+1.4), L18 (+1.3) and L15 (+0.9) mixers and the L18, L19 and L23 MLPs (+1.2 to +1.4 each), while L21 recall declines (-2.0). The alternatives shift over the repeats, so these numbers are rough. No single component carries the copy, and the 19.5 test showed that removing one route gets compensated. Further head surgery on the copy routes therefore looks low-yield. The larger remaining GSM problem is over-doubting after solving. To see what triggers it, I'll read two GSM failures from the current edit in which the correct value appeared early: the problem, the context where the answer first appears, and the following lines, where the model starts re-examining.

```python
# Hypothesis: after reaching the right answer, C's GSM failures enter re-examination via specific line-start phrases
def show_after_answer(x, nafter=900):
    t=x['text']; p=first_num_pos(t,x['ans'])
    print('==',x['id'],'ans',x['ans'],'ok',x['ok'],'cut',x['cut'],'tokens',x['tokens'],'first ans at char',p,'of',len(t))
    print(PROB[x['id']]['problem'][:300].encode('ascii','replace').decode())
    print('--- after answer:'); print(t[max(0,p-150):p+nafter].encode('ascii','replace').decode())
fails=[x for x in allC if x['src']=='gsm8k' and not x['ok'] and (first_num_pos(x['text'],x['ans']) or -1)>=0]
print(len(fails))
show_after_answer(fails[3], 1100)
```