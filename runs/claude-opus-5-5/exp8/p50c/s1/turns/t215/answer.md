'Ready' turns out to be rare (0.011 per 1k tokens under C). It precedes </think> in only about 13% of C's endings, so its suppression by the brake is not a meaningful obstacle. The ending ritual is mostly "Final Answer" / "Final check" / "answer is". The undocumented gen_21 variant had more 'Ready' endings but the same accuracy. So token-level side effects of the brake are not the bottleneck; I am dropping that line. By rough arithmetic, the most valuable remaining target is the commit failure: about a third of C's cut answers (≈8% of all answers) had already reached the correct answer. Remaining loops on unsolved hard problems would convert to correct answers far less often. I will measure, from the texts, how the correct-but-cut answers differ from the correct-and-finished ones after the first correct concluding statement: how long until </think> versus what follows instead (re-verification, doubt or a switch to a different answer).

```python
# Commit failure anatomy under C: after the first concluding statement of the CORRECT answer, what happens (ok vs cut)?
def norm_ans(a): return re.sub(r'\s+', '', str(a)).replace('\\!', '').replace('$', '')
def first_correct_stmt(t, ans):
    a = re.escape(norm_ans(ans))
    pats = [rf'answer is\s*:?\s*\$?\\?(?:boxed\{{)?\s*{a}', rf'\\boxed\{{\s*{a}\s*\}}', rf'(?:Result|Answer|Total|Final Answer)\s*[:=]\s*\$?\s*{a}\b', rf'=\s*\$?{a}\$?\s*\.?\s*\n']
    best = None
    for p in pats:
        m = re.search(p, t.replace(' ', ' '))
        if m and (best is None or m.start() < best): best = m.start()
    return best
stats = {'ok': [], 'cut': []}
for name, g, ids in [('S1', g13, S1), ('H1', g16e, H1)]:
    for i in ids:
        s = g[i]; t = s['text']; ans = None
        ans = [r for r in rows if r['id'] == i][0]['answer'] if name == 'S1' else None
        stats.setdefault('_', 0)
print('need answers for H1 ->', type(g16e[H1[0]]))
```