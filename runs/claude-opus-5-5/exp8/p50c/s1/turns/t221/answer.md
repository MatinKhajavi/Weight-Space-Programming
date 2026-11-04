Each indentation depth is a separate single token (for example 3 spaces = id 262, 59 spaces = id 11611, 79 spaces = id 38177), and deeper ones are rarer tokens. Under the presence penalty, every depth already used gets -1.5 while unused deeper depths get nothing, so the sampler systematically favours going deeper. This drift could also interact with our edit. The ablated induction heads likely copied the previous line's indentation at each newline. Removing them may have weakened indentation consistency, and C's loops indeed sit at 32–80 spaces whereas the original's loops sat at 4. Before treating this as a mechanism, I need to know whether depth relates to outcome and to the edit. I will compare the maximum indentation reached within the first 100/200/400 lines across the original, A and C on S1, and within C between correct and failed answers by source.

```python
# Does indentation drift depend on the edit, and does early depth predict failure? (max indent within first N lines)
def depth_profile(t, Ns=(100, 200, 400)):
    k = t.find('</think>'); th = t[:k] if k > 0 else t
    lines = [l for l in th.split('\n') if l.strip()]
    ind = [len(l) - len(l.lstrip(' ')) for l in lines]
    return [max(ind[:N]) if len(ind) >= N else np.nan for N in Ns]
for tag, g in [('orig', gO), ('A', gA), ('C', g13)]:
    P_ = np.array([depth_profile(g[i]['text']) for i in S1])
    print(tag, 'median max-indent at 100/200/400 lines', np.nanmedian(P_, 0), '| share >=24 spaces by 200 lines', np.round(np.nanmean(P_[:, 1] >= 24), 2))
for src in ['gsm', 'math']:
    for st in ['ok', 'fail']:
        vals = []
        for name, g, ids in [('S1', g13, S1), ('H1', g16e, H1)]:
            for i in ids:
                if (i.startswith('gsm')) != (src == 'gsm'): continue
                ok = str(g[i]['correct']) == 'True'
                if (st == 'ok') == ok: vals.append(depth_profile(g[i]['text']))
        V = np.array(vals); print('C', src, st, len(V), 'median max-indent 100/200 lines', np.nanmedian(V[:, :2], 0), 'n with >=100 lines', int(np.sum(~np.isnan(V[:, 0]))),
                                 'share >=24 @100 lines', np.round(np.nanmean(V[:, 0] >= 24), 2))
```