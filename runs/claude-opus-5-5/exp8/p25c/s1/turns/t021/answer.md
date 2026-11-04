Separation between finished and looping samples is near-perfect at every layer from 2 onward, but this is uninformative: the averaged window of looping samples is full of loop text while finished ones include the final answer, so any content difference separates them. Contrastive averages therefore cannot localize the mechanism; I need token-level comparisons where content is held fixed. Before designing that, I want to test a second hypothesis about why the model never finishes: perhaps it reaches natural closing points (e.g., "Done thinking.", "I will write the solution now.") but fails to emit </think>, keeps going, and then falls into a loop. I'll search the looping texts for such wrap-up phrases and see what follows them, compared with the finished texts where they are followed by </think>. If wrap-up phrases in looping texts are frequently followed by more thinking rather than </think>, a suppressed end-of-thinking transition is implicated; if they rarely appear, the loops themselves are the primary problem.

```python
# Do looping texts reach wrap-up phrases without closing thinking?
phr = ['Done thinking', 'ready to generate', 'write the solution now', 'write out the solution', 'Final Answer', 'final answer is', "I'm confident", 'I will now write', 'Ready to write', 'Final check']
def count_phr(texts):
    return {p: np.mean([p.lower() in t.lower() for t in texts]) for p in phr}
fin_txt = [t.split('</think>')[0] for t in fin.text]
cut_txt = list(cut.text)
a = count_phr(fin_txt); b = count_phr(cut_txt)
for p in phr: print('%-25s fin %.2f  cut %.2f' % (p, a[p], b[p]))
# what follows 'Done thinking' in cut texts
cnt = collections.Counter()
for t in cut_txt:
    for m in re.finditer(r'Done thinking\.?', t):
        cnt[t[m.end():m.end()+15].replace('\n','|')] += 1
print(cnt.most_common(8))
```