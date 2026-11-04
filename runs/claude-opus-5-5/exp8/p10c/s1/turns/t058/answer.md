The endings confirm the picture. Finished runs leave the thinking through an explicit concluding line ("Okay, I'm confident.", "The answer is 100.", "I will write the response now.", "Final check ..."), chosen at a line start. Looping runs keep choosing a doubting line ("Wait, maybe ...") at those same decision points. So the key decision is "doubt again vs. conclude" at line and bullet starts. Next, I measure this decision directly in the two loop texts I have per-token data for. At every line-start position I compute P(" Wait") from the top-20 log-probs, the probability of typical concluding first words, and how these evolve along the text and across loop cycles. If self-reinforcement drives the failure, P(Wait) at line starts should climb over the text and approach 1 inside the loop, with concluding words squeezed out. If P(Wait) stays moderate and the loop is held mostly by other tokens, I should target the copying lock-in rather than the doubting decision.

```python
# Hypothesis: P(Wait) at line-start decisions rises along the text and saturates in the loop
def line_start_positions(ids):
    pos = []; 
    for t in range(1, len(ids)-1):
        nxt = tok.decode([ids[t+1]])
        if not nxt.strip() or nxt.strip() in ('*','-'): continue
        # tokens from last newline to t must be whitespace / bullet markers
        k = t; ok = False
        while k >= 0:
            s = tok.decode([ids[k]])
            if '\n' in s: ok = s.split('\n')[-1].strip(' *-') == ''; break
            if s.strip(' *-') != '': break
            k -= 1
        if ok: pos.append(t)
    return pos
CONC = [tok.encode(w).ids[0] for w in [' Okay', ' The', ' Final', ' I', ' So', ' This', 'Okay', 'The', 'Final', 'So']]
for j in range(2):
    ids,p,on = info[j]
    ls = line_start_positions(ids)
    pw = []; pc = []
    for t in ls:
        d = dict(zip(ti[j,t].tolist(), np.exp(tl[j,t]).tolist()))
        pw.append(sum(d.get(w,0) for w in WAIT)); pc.append(sum(d.get(w,0) for w in CONC))
    pw = np.array(pw); pc = np.array(pc); ls = np.array(ls)
    pre = ls < on
    q = np.array_split(np.arange(pre.sum()), 4)
    print(f'T{j}: n line starts pre {pre.sum()} loop {(~pre).sum()} | P(Wait) by quarter pre:', [round(float(pw[pre][x].mean()),3) for x in q],
          'loop:', round(float(pw[~pre].mean()),3), '| P(conclude) pre:', [round(float(pc[pre][x].mean()),3) for x in q], 'loop:', round(float(pc[~pre].mean()),3))
    print('   actual next=Wait at line starts pre:', int(sum(ids[t+1] in WAIT for t in ls[pre])), 'loop:', int(sum(ids[t+1] in WAIT for t in ls[~pre])))
```