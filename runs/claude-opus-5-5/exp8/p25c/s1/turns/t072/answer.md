E3 results are in. Mean log-probabilities per text are similar to the original (e.g., −0.716 vs the E1/E2 range on text 0), so nothing broke grossly. Now I check:
1. Did reversing the direct "Wait" push flatten P("Wait") at line starts, and raise per-period escape?
2. Non-loop token log-probabilities (collateral-damage proxy).
3. A layer-by-layer decomposition of the "Wait" margin at the decision points, E3 versus original.

The decomposition shows where the compensation sits: if later MLPs (layers 20–23) or attention in layers 20–23 push "Wait" back up under E3, those are the self-repair pathways a robust fix must also address.

```python
# Test E3: P(Wait) at line starts, escape prob, non-loop damage, and which components compensate (margin decomposition E3 vs orig)
F10 = '/work/fwd_10'
lp10 = np.load(f'{F10}/token_logprobs.npy'); tkl10 = np.load(f'{F10}/topk_logprobs.npy'); tki10 = np.load(f'{F10}/topk_ids.npy')
rows=[]
for b_, t_, k_, nxt_ in dp_keys:
    for nm_, lp_, tl_, ti_ in [('orig', lp4, tkl, tki), ('E2', lp9, tkl9, tki9), ('E3', lp10, tkl10, tki10)]:
        ps_ = samp_probs(tl_[b_,t_], ti_[b_,t_], None); m_ = np.where(ti_[b_,t_]==ids4[b_,t_+1])[0]
        rows.append(dict(model=nm_, k=k_, isWait=(nxt_=='Wait'), p=float(np.exp(lp_[b_,t_])), ps=float(ps_[m_[0]]) if len(m_) else 0.0))
DQ3 = pd.DataFrame(rows)
print('P(Wait) at line starts:'); print(DQ3[DQ3.isWait].groupby(['model','k']).p.mean().unstack('k').round(3).to_string())
A3 = analyze(lp10, tkl10, tki10, False)
print(pd.DataFrame({'p_orig':A0[A0.k>0].groupby('k').p.mean(),'p_E3':A3[A3.k>0].groupby('k').p.mean(),'esc_orig':esc(A0),'esc_E2':esc(A2),'esc_E3':esc(A3)}).round(3).to_string())
print('non-loop dlogprob E3:', round(np.mean([lp10[b_,t_]-lp4[b_,t_] for b_,t_ in lead_mask]),4))
# decomposition at Wait decision points
HSe = {l_: np.load(f'{F10}/hidden_{l_}.npy', mmap_mode='r') for l_ in range(25)}
MAe = {l_: np.load(f'{F10}/mlp_act_{l_}.npy', mmap_mode='r') for l_ in range(15,24)}
recs=[]
for b_, t_, k_ in dp_orig:
    rms_o = np.sqrt((HS[24][b_,t_]**2).mean()+1e-6); rms_e = np.sqrt((HSe[24][b_,t_]**2).mean()+1e-6)
    for l_ in range(24):
        for nm_, H_, M_, rms_ in [('orig', HS, MA, rms_o), ('E3', HSe, MAe, rms_e)]:
            tot_ = H_[l_+1][b_,t_] - H_[l_][b_,t_]
            if l_ >= 15:
                mlp_ = M_[l_][b_,t_] @ Wd_all[l_].T; att_ = tot_ - mlp_
                recs.append(dict(model=nm_, k=k_, comp=f'{l_}a', v=float((att_*gFn/rms_)@uW))); recs.append(dict(model=nm_, k=k_, comp=f'{l_}m', v=float((mlp_*gFn/rms_)@uW)))
            else:
                recs.append(dict(model=nm_, k=k_, comp='0-14', v=float((tot_*gFn/rms_)@uW)))
R3 = pd.DataFrame(recs)
T3 = R3.groupby(['comp','model','k']).v.mean().unstack(['model','k'])
summ3 = pd.DataFrame({'orig_k1':T3[('orig',1)], 'E3_k1':T3[('E3',1)], 'orig_k6':T3[('orig',6)], 'E3_k6':T3[('E3',6)]})
summ3['chg_k6'] = summ3.E3_k6 - summ3.orig_k6
print(summ3.sort_values('chg_k6', key=abs, ascending=False).head(12).round(2).to_string())
print('total margin orig/E3 at k=1,6:', [round(float(R3[(R3.model==m_)&(R3.k==k_)].groupby(['k']).v.sum().iloc[0]/len(dp_orig[dp_orig[:,2]==k_])),2) for m_ in ['orig','E3'] for k_ in [1,6]])
```