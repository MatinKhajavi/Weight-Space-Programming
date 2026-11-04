Brake-only leaves 138 answers cut, 127 of them exact loops, while "Wait" is gone. So the verbatim induction loops are a separate failure that the brake doesn't touch. Adding the head ablation on top of the brake adds +15.2 points (SE 3.0). The remaining loops under the combined edit come in three kinds. Some are short in-line computational stutters, such as "Faces = 2×2 + 3×3 =" with a 52-character period. Some are long reasoning-level cycles (529 and 1,957 characters). And one is a distinct pattern: the model has boxed 196 and repeats "Okay, I'll stop. \boxed{196} Final Answer:" without ever emitting "</think>". That last kind matters because the brake neuron's output direction also carries a small positive "</think>" component (z = 2.2). Its negative activation at line starts, now doubled, could be suppressing the end-of-thinking token as a side effect. I'll count how many cut answers under the combined edit are "finished but can't close" loops, with \boxed or "final answer" inside the repeating unit. I'll check whether the grader gives credit for a box inside an unfinished thinking (pred is None or not). And I'll compute how many logits the doubled brake removes from "</think>" relative to the "Wait" suppression. If such loops are common and the "</think>" side effect is non-trivial, a refined brake that leaves the "</think>" component unchanged would be a principled improvement.

```python
# Check: 'finished-but-cannot-close' loops under A+brake, grading of cut samples, and brake's </think> side effect
fa_loops = []
for i, p_ in zip(cutC, pers):
    t = gC[i]['text']; unit = t[-p_:] if p_ else t[-400:]
    if ('\\boxed' in unit) or re.search(r'(?i)final answer|I\'ll stop|done', unit): fa_loops.append(i)
print('A+brake cut=%d; final-answer-type loops=%d; their pred values: %s' % (len(cutC), len(fa_loops), [gC[i]['pred'] for i in fa_loops][:8]))
print('any cut sample with non-None pred:', sum(gC[i]['pred'] not in (None,'None') for i in cutC))
same_in_A = [i for i in S1 if gA[i]['cut'] and re.search(r'\\boxed', gA[i]['text'][-2000:])]
print('A cut with boxed in tail:', len(same_in_A))
# brake column logit effect: </think> vs alternatives, relative to Wait
col0 = load_original(name22)[:,573].numpy()*(1.0+wn.numpy())
for nmx, tkn in (('Wait', WAIT), ('</think>', 248069), ('Hmm', tid(' Hmm')), ('So', tid(' So')), ('Okay', tid(' Okay')), ('Final', tid(' Final')), ('**', tid('**'))):
    print('%-9s col.E = %+.3f' % (nmx, col0 @ Eemb[tkn]), end=' | ')
print()
```