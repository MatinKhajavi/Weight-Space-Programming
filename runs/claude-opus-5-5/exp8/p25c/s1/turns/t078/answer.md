The "other" loops are mostly short "So" restatement loops ("So $\sin\theta = 1/5$." repeated), single-line "If he keeps…" loops, and runs of numeric lines. After E3, "So" has taken over the role "Wait" played: the runaway of copied line openers moved to the next-best opener.

This suggests generalizing E3 from the single "Wait" direction to the subspace that tells common line openers apart from each other. The directions are centred on the openers' mean, so the model's tendency to start a new line or concluding statement is untouched; only "repeat the same opener because earlier lines used it" is reversed. Before building it, I will count which line openers actually occur in the remaining loops and in finished reasoning, to pick the set from data and avoid hand-picking.

```python
# Choose the opener set from data: most frequent line-opening tokens inside E3 loops vs finished thinking
def opener_tokens(text):
    out_=[]
    for ln_ in text.split('\n'):
        s_ = re.sub(r'^[\s*\-\d.)]*', '', ln_)
        if not s_: continue
        ids_ = tok.encode(s_).ids
        if ids_: out_.append(ids_[0])
    return out_
cl = Counter(); cf = Counter()
for _, r in LS11[LS11.loop].iterrows():
    s_ = G11[r['key']]; P_ = int(r['P']); cl.update(set(opener_tokens(s_['text'][-P_:])))
for s_ in fin11: cf.update(opener_tokens(s_['text'].split('</think>')[0]))
nloop = int(LS11.loop.sum()); nfin = sum(cf.values())
print('top openers in E3 loops (share of loops containing):')
print([(tstr(i_), round(c_/nloop,2)) for i_, c_ in cl.most_common(25)])
print('top openers in finished thinking (share of lines):')
print([(tstr(i_), round(c_/nfin,3)) for i_, c_ in cf.most_common(25)])
```