The teacher-forced pass returned all 11 texts: 6 wrap-up loops and 5 real closes. I now read P(`</think>`) (id 248069) at every line-end position from the top-20 log-probabilities. If it is not among the top 20, I record the 20th probability as an upper bound. For the closing texts the key number is P(`</think>`) at the actual close and at the preceding line ends. For the wrap-up loops I track it across successive repetitions of the wrap-up line, along with the main competitors at those points.

```python
# P(</think>) at line ends: real closes vs wrap-up loops (by repetition), from top-20 logprobs (fwd_15)
F15 = '/work/fwd_15/'
ids15 = np.load(F15+'input_ids.npy'); lp15 = np.load(F15+'token_logprobs.npy')
tki15 = np.load(F15+'topk_ids.npy'); tkl15 = np.load(F15+'topk_logprobs.npy')
lens15 = [199,291,280,294,178,497,178,189,155,153,152]
END = 248069
def p_end(i, t):
    m = np.where(tki15[i, t] == END)[0]
    if len(m): return float(np.exp(tkl15[i, t, m[0]])), True
    return float(np.exp(tkl15[i, t, -1])), False      # upper bound
LE = []   # line-end records
for i in range(11):
    L_ = lens15[i]; toks = ids15[i, :L_]
    st = int(np.where(toks == think_id)[0][-1]) + 1
    strs = [tok.decode([int(x)]) for x in toks]
    seen_lines = collections.Counter(); line_start = st
    for p in range(st, L_-1):
        if '\n' in strs[p]:
            line = tuple(int(x) for x in toks[line_start:p+1]); line_txt = ''.join(strs[line_start:p+1]).strip()
            k = seen_lines[line_txt]; seen_lines[line_txt] += 1
            pe, exact = p_end(i, p)
            top3 = [(repr(tok.decode([int(a)]))[:14], round(float(np.exp(b)),2)) for a, b in zip(tki15[i, p][:3], tkl15[i, p][:3])]
            LE.append(dict(i=i, kind='close' if i>=6 else 'wrap', p=p, k=k, pend=pe, exact=exact, line=line_txt[:40], top3=top3, nxt=int(toks[p+1])))
            line_start = p+1
LE = pd.DataFrame(LE)
for i in range(11):
    sub = LE[LE.i==i]
    s = ' '.join([f"{r.pend:.3f}{'' if r.exact else '<'}" for r in sub.itertuples()])
    print(i, 'wrap' if i<6 else 'close', '| P(end) at successive line ends:', s[:300])
print(LE[LE.kind=='close'].groupby('i').tail(1)[['i','line','pend','top3']].to_string())
```