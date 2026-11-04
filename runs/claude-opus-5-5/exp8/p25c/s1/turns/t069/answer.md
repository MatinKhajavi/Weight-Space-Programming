Now I compare E2 (the "Wait" readout removed from layer-15/19 attention) with the original and with E1 on the same 14 loop texts:
- P(actual opener) at the line-start decision points, by repetition,
- the probability of the actual loop token over all positions,
- the per-period escape probability under the evaluation sampler,
- the log-probability change on non-loop tokens (prompt and lead), as a crude collateral-damage check.

The hypothesis predicts a large drop in P("Wait") at line starts and much flatter growth with repetition. I note that the mean log-probabilities of the texts are slightly *higher* than for E1 (e.g., −0.677 vs −0.701 on text 0), so the overall fit did not degrade.

```python
# Test E2: does removing the Wait readout from L15/L19 attention lower P(Wait) at line starts and raise escape probability?
F9 = '/work/fwd_9'
lp9 = np.load(f'{F9}/token_logprobs.npy'); tkl9 = np.load(f'{F9}/topk_logprobs.npy'); tki9 = np.load(f'{F9}/topk_ids.npy')
assert np.array_equal(np.load(f'{F9}/input_ids.npy'), ids4)
dp_keys = DP[DP.model=='orig'][['b','t','k','nxt']].values
rows=[]
for b_, t_, k_, nxt_ in dp_keys:
    for nm_, lp_, tl_, ti_ in [('orig', lp4, tkl, tki), ('E1', lp7, tkl7, tki7), ('E2', lp9, tkl9, tki9)]:
        ps_ = samp_probs(tl_[b_,t_], ti_[b_,t_], None)
        m_ = np.where(ti_[b_,t_]==ids4[b_,t_+1])[0]
        rows.append(dict(model=nm_, k=k_, isWait=(nxt_=='Wait'), p=float(np.exp(lp_[b_,t_])), ps=float(ps_[m_[0]]) if len(m_) else 0.0))
DQ = pd.DataFrame(rows)
print('P(actual opener) at line starts (Wait openers only):'); print(DQ[DQ.isWait].groupby(['model','k']).p.mean().unstack('k').round(3).to_string())
print('sampler prob of continuing with same opener:'); print(DQ[DQ.isWait].groupby(['model','k']).ps.mean().unstack('k').round(3).to_string())
A2 = analyze(lp9, tkl9, tki9, False)
A0 = analyze(lp4, tkl, tki, False); A1 = analyze(lp7, tkl7, tki7, False)
def esc(A_): return A_[A_.k>0].groupby(['b','k']).ps.apply(lambda s: 1-np.prod(s)).groupby('k').mean()
print(pd.DataFrame({'p_orig':A0[A0.k>0].groupby('k').p.mean(),'p_E2':A2[A2.k>0].groupby('k').p.mean(),'esc_orig':esc(A0),'esc_E1':esc(A1),'esc_E2':esc(A2)}).round(3).to_string())
lead_mask = [(b_, t_) for b_ in range(14) for t_ in range(L4[b_]-1) if (np.asarray(REP[b_])[t_+1] if t_+1 < len(REP[b_]) else 0)==0]
dl_E1 = np.mean([lp7[b_,t_]-lp4[b_,t_] for b_,t_ in lead_mask]); dl_E2 = np.mean([lp9[b_,t_]-lp4[b_,t_] for b_,t_ in lead_mask])
print('non-loop tokens (prompt+lead) mean dlogprob: E1', round(dl_E1,4), 'E2', round(dl_E2,4))
```