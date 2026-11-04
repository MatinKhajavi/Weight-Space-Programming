That run was wasted on bookkeeping: the H1 reference answers were not at hand. I will load all reference answers from the practice files and redo the commit-failure analysis properly. For each C answer on S1 and H1 I will find the first concluding statement of the correct answer ("answer is X", \boxed{X}, "Final Answer: X", "Result: X"). For finished answers I measure the distance to </think>. For cut answers I measure how far into the text the solution came, how often it was restated, and whether other final answers were entertained afterwards. The hypothesis is that failed commits are re-verification and doubt spirals around a correct answer rather than a switch to a wrong one.

```python
# Commit-failure anatomy under C (retry with reference answers loaded from practice files)
ref = {}
for f in ['/practice/gsm8k_train.jsonl', '/practice/math_train.jsonl']:
    for l in open(f):
        r = json.loads(l); ref[r['id']] = r['answer']
def nows(s): return re.sub(r'\s+', '', s)
def concl_positions(t, ans):
    T = nows(t); a = re.escape(nows(str(ans)))
    pat = rf'(?:answeris:?\$?(?:\\boxed\{{)?|\\boxed\{{|FinalAnswer:?\$?(?:\\boxed\{{)?|Result:\$?|Answer:\$?){a}(?![0-9])'
    return [m.start() / max(1, len(T)) for m in re.finditer(pat, T)], T
def other_answers(T, ans, after):
    a = nows(str(ans)); out = set()
    for m in re.finditer(r'(?:answeris:?\$?(?:\\boxed\{)?|\\boxed\{|FinalAnswer:?\$?(?:\\boxed\{)?)([^\s\$\}\.,]{1,20})', T[int(after * len(T)):]):
        if m.group(1) != a: out.add(m.group(1))
    return out
ok_gap = []; cut_info = []
for name, g, ids in [('S1', g13, S1), ('H1', g16e, H1)]:
    for i in ids:
        s = g[i]; t = s['text']; ans = ref[i]; ntok = int(s['tokens'])
        pos, T = concl_positions(t, ans)
        if str(s['correct']) == 'True':
            k = nows(t).find('</think>')
            if pos and k > 0: ok_gap.append((k / len(T) - pos[0]) * ntok)
        elif str(s['cut']) == 'True':
            cut_info.append((i, pos[0] * ntok if pos else None, len(pos), len(other_answers(T, ans, pos[0])) if pos else None))
ok_gap = np.array(ok_gap); print('C-ok: first correct concl stmt -> </think> tokens: quantiles', np.percentile(ok_gap, [10, 25, 50, 75, 90]).round(), 'n', len(ok_gap))
solved = [c for c in cut_info if c[1] is not None]
print('C-cut', len(cut_info), 'with correct concluding stmt', len(solved))
print('  first stmt token pos quantiles', np.percentile([c[1] for c in solved], [10, 25, 50, 75, 90]).round())
print('  number of correct concl stmts quantiles', np.percentile([c[2] for c in solved], [10, 50, 90]), '| other final answers after: share>0', np.mean([c[3] > 0 for c in solved]).round(2))
```