Removing the brake's readiness suppression has no effect: the paired difference V2 − C is +0.39 points (SE 2.32), with 18 wrong→right and 17 right→wrong flips. "Ready" lines were essentially absent in both (0.01 per 1,000 tokens), and correct-answer length did not change. The side effect was real in the weights but irrelevant in practice, because this model rarely concludes via "Ready". I will revert to the simpler, held-out-confirmed edit C. The run also shows that near-identical models flip about 35 of 256 answers by sampling chaos alone. Edits gaining less than about 5 points are therefore undetectable at this sample size, and I should look for larger levers. Next I check the finished-but-wrong answers, 28 to 47 per 256 under C, for a systematic cause: formatting mismatches such as units or text inside \boxed{} that the grader rejects, truncated answers, or genuine reasoning errors. A formatting problem would be a distinct failure mechanism with its own fix. Genuine errors would point back to the cut answers as the remaining lever.

```python
# Revert to C; then inspect C's wrong (finished) answers: pred vs reference, to separate formatting mismatches from reasoning errors
W22C=W22o.clone(); W22C[:,573]=2*c573; save_edit(name22, W22C)
print('edited:', edited(), 'Wait eff', eff_tokens(load(name22)[:,573])[' Wait'])
wr_rows=[]
for (runn,r_,s_) in allrecs:
    if not s_['correct'] and not s_['cut']:
        wr_rows.append((runn, r_['id'][:28], r_['answer'][:25], str(s_['pred'])[:40], s_['tokens']))
print(len(wr_rows))
for w_ in wr_rows[:75]: print(w_)
```