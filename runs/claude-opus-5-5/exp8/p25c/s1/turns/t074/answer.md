E3 brings a large improvement on the diagnosis set. Accuracy rises from 10.4% to 27.0%: GSM8K 14.6%→32.3%, MATH 7.8%→23.8%. The cut-off share drops from 90% to 73%, and only 3 of the 141 finished answers are wrong. So stopping the copy-driven runaway of "Wait" lets many more problems finish, and the extra finished answers are almost all correct. I now need three things:
1. The paired difference against gen_2 with a problem-clustered standard error.
2. What the remaining 73% of cut-off answers look like: still exact loops? With which line openers, starting when?
3. Whether the edit changed normal reasoning, e.g., length of finished answers, and the "Wait" share in finished text.

If the remaining loops mostly use other openers or repeat within lines, the next refinement should target that residual copying. These numbers come from the set used to develop the edit, so a fresh-problem confirmation will follow.

```python
# Analyze E3 generations: paired diff vs orig, remaining loop structure (openers, onset), finished-answer stats
G11 = load_gen('/work/gen_11.jsonl')
def paired(Ga, Gb):
    ks = sorted(set(Ga)&set(Gb)); d = np.array([int(Gb[k]['correct'])-int(Ga[k]['correct']) for k in ks])
    byp_ = defaultdict(list)
    for k, v in zip(ks, d): byp_[k[0]].append(v)
    pm_ = np.array([np.mean(v) for v in byp_.values()])
    return d.mean()*100, pm_.std(ddof=1)/np.sqrt(len(pm_))*100, int((d>0).sum()), int((d<0).sum())
print('E3 vs orig: diff %.2f pts, SE %.2f, w->r %d, r->w %d' % paired(G2, G11))
print('E3 vs E1  : diff %.2f pts, SE %.2f, w->r %d, r->w %d' % paired(G8, G11))
for src_ in ['gsm8k','math']:
    sub2 = {k:v for k,v in G2.items() if v['src']==src_}; sub11 = {k:v for k,v in G11.items() if v['src']==src_}
    print(src_, 'diff %.2f SE %.2f w->r %d r->w %d' % paired(sub2, sub11))
LS11 = loop_stats(G11)
print('E3 cut', len(LS11), 'exact-loop share', round(LS11.loop.mean(),3), 'onset median', int(LS11.onset_tok.median()), 'IQR', LS11.onset_tok.quantile([.25,.75]).astype(int).tolist(), 'period median', int(LS11.P.median()))
def loop_opener(text, P_, st_):
    per = text[st_:st_+P_]
    lines_ = [l_.strip().lstrip('*').strip() for l_ in per.split('\n') if l_.strip()]
    ws = [re.match(r'[A-Za-z]+', l_).group(0) if re.match(r'[A-Za-z]+', l_) else 'nonword' for l_ in lines_]
    return ws
for nm_, G_, LS_ in [('orig', G2, LS2), ('E3', G11, LS11)]:
    cnt_ = Counter()
    for _, r in LS_[LS_.loop].iterrows():
        s_ = G_[r['key']]; ws = loop_opener(s_['text'], int(r['P']), len(s_['text'])-int(r['P'])*3)
        cnt_.update(set(ws))
    n_loop = int(LS_.loop.sum())
    print(nm_, 'loops containing opener (share):', {w_: round(c_/n_loop,2) for w_, c_ in cnt_.most_common(8) if w_ in WL or w_=='nonword'})
fin11 = [s_ for s_ in G11.values() if not s_['cut']]
print('finished: n', len(fin11), 'acc among finished', round(np.mean([s_['correct'] for s_ in fin11]),3), 'median tokens', int(np.median([s_['tokens'] for s_ in fin11])))
print('Wait-opener share in finished thinking: orig', round(np.mean([opener_share(s_['text'].split('</think>')[0]) for s_ in G2.values() if not s_['cut']]),3), 'E3', round(np.mean([opener_share(s_['text'].split('</think>')[0]) for s_ in fin11]),3))
```