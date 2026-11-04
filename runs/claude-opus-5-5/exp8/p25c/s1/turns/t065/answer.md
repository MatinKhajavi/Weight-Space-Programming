E1 does not change the loop picture in real generations:
- The paired accuracy gain is +1.95 ± 1.89 points (not significant), with 44 answers flipping wrong→right and 34 right→wrong.
- 96.5% of cut-off answers still end in exact loops.
- The median loop onset is unchanged (6,719 vs 6,712 tokens); only the upper quartile moved later and periods got longer.

So the extra escape chance seen in short texts does not survive hundreds of periods of real looping. The model has redundant ways to predict loop tokens, and the per-period decision is made at a few line-start positions.

Reading the compact loops again, most are verification loops after the right answer was already found: "Wait, $135\times6=810$" (answer 810), "$15\times1.4=21$" (21), "$7.8\times60=468$" (468), "$n=12$" (12). The model re-checks the same step instead of concluding.

I therefore want to look precisely at the line-start decision points. I will check how the probability of "Wait" versus concluding alternatives ("So", "Okay", "Therefore", "</think>") changes with repetition, in the original and with E1. If "Wait" at line starts climbs toward certainty with repetition while the concluding options are pushed out of the top-p window, then the lock-in acts on that choice. The fix should then target what copies the line opener and line content.

```python
# Hypothesis: lock-in happens at line starts: P('Wait') rises with k and concluding alternatives fall below the top-p window
WL = {'Wait','So','Let','If','This','Okay','The','Therefore','Thus','Hence','Yes','No','Actually','But','And','Hmm','I','We','Final','Answer','</think>','Is','Now','Check','Since','Then','Alternatively','To','Also','In','Done','Hold','OK','Correct','Good','Great','Right','Hmm,'}
def tstr(i_):
    s_ = tok.decode([int(i_)]).strip()
    return s_ if s_ in WL or re.fullmatch(r'[0-9$\\{}()*.,:=+\-^_/\s]{1,6}', s_ or ' ') else f'#{int(i_)}'
nl_ids = set(i_ for i_ in range(len(tok.get_vocab())) if False)
rows=[]
for b_ in range(14):
    n_ = L4[b_]; rep_ = np.asarray(REP[b_]); ids_ = ids4[b_]
    for t_ in range(n_-1):
        k_ = rep_[t_+1] if t_+1 < len(rep_) else 0
        if k_ == 0: continue
        prev = tok.decode([int(ids_[t_])])
        if not prev.endswith('\n') and not (prev.strip()=='' or prev.strip() in ('*',)) : continue
        nxt_s = tstr(ids_[t_+1])
        if not re.fullmatch(r'[A-Za-z<>/]+', tok.decode([int(ids_[t_+1])]).strip() or '#'): continue
        for nm_, tl_, ti_ in [('orig', tkl, tki), ('E1', tkl7, tki7)]:
            pr_ = np.exp(tl_[b_,t_]); top = [(tstr(ti_[b_,t_,q_]), round(float(pr_[q_]),3)) for q_ in range(4)]
            rows.append(dict(b=b_, t=t_, k=k_, model=nm_, nxt=nxt_s, p_nxt=float(np.exp((lp4 if nm_=='orig' else lp7)[b_,t_])), top=top))
DP = pd.DataFrame(rows)
print('line-start decision points per model:', DP.groupby('model').size().to_dict())
print(DP.groupby(['model','k']).p_nxt.mean().unstack('k').round(3).to_string())
print('opener counts (orig):', DP[DP.model=='orig'].nxt.value_counts().head(8).to_dict())
for b_ in [1, 8, 10]:
    sub_ = DP[(DP.b==b_)&(DP.model=='orig')]
    for _, r in sub_.iterrows(): print(b_, r.k, r.nxt, round(r.p_nxt,3), r.top[:3])
```