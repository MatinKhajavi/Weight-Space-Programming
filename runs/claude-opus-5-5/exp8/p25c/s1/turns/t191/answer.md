The verification supports the account of the final edit, with numbers close to the earlier E3 measurements.
- **Escape from loops:** in the original model, the chance of breaking out within a period collapses as a loop repeats (0.29 at the third repetition, 0.07 at the fourth, 0.006 at the fifth). Under E6 it stays high: 0.85, 0.88 and 0.91 at the same repetitions. That is stronger than E3 (about 0.71–0.75). Removing the opener copying in L15/L19 only (E2) changes almost nothing (0.08 at the fourth repetition).
- **Repeated-token probability:** averaged over all loop tokens, the probability of the repeated token falls only modestly (0.991 → 0.942 at the fifth repetition). That fits an edit acting at decision points rather than on every copied token.
- **Specificity:** on the 1,679 normal-reasoning lead tokens, the mean log-probability change is −0.018 and the median 0.000, with only 3% of tokens changing by more than one nat.

Next I'll check that those 3% are mostly the targeted line-opener decisions rather than math content. I'll also measure P(actual opener) and P(Wait) at line-start decision points by repetition, original vs E6. Earlier, P(Wait) at line starts was 0.57–0.98 in the original and 0.23–0.13 under E3.

```python
# Which lead tokens change most under E6 (openers vs content)? and P(opener)/P(Wait) at line-start decision points by repetition k
LPo=np.load('/work/fwd_4/token_logprobs.npy'); LP6=np.load('/work/fwd_25/token_logprobs.npy')
TKI6=np.load('/work/fwd_25/topk_ids.npy'); TKL6=np.load('/work/fwd_25/topk_logprobs.npy'); TKIo=np.load('/work/fwd_4/topk_ids.npy'); TKLo=np.load('/work/fwd_4/topk_logprobs.npy')
wait_ids=set(tok.encode('Wait').ids+tok.encode(' Wait').ids); print('Wait ids', wait_ids)
OPW=['Wait','So','Let','Okay','If','I','This','But']
opener_ids={i for w in OPW for i in tok.encode(w).ids+tok.encode(' '+w).ids if len(tok.encode(w).ids)==1}
big=Counter(); bigop=0; nbig=0; rows_ls=[]
for b in range(14):
    T=L4[b]; rk=rep_index(b); ids=ids4[b,:T]
    for t in range(T-1):
        y=ids[t+1]; k=rk[t+1]; d=LP6[b,t]-LPo[b,t]
        prev=tok.decode([ids[t]]); ls=prev.endswith('\n') or (prev.strip()=='' and t>0 and tok.decode([ids[t-1]]).endswith('\n')) or prev.strip() in ('*',)
        if k==0 and rk[t]==0 and t>=T-6*0-1-1-200 and abs(d)>1:
            nbig+=1; bigop+= y in opener_ids; big[tok.decode([y]).strip()[:10]]+=1
        if ls and k>=1: rows_ls.append((k, y in opener_ids, y in wait_ids, np.exp(LPo[b,t]), np.exp(LP6[b,t]),
                                         sum(np.exp(TKLo[b,t,j]) for j in range(20) if TKIo[b,t,j] in wait_ids), sum(np.exp(TKL6[b,t,j]) for j in range(20) if TKI6[b,t,j] in wait_ids)))
print('lead+prompt tokens |d|>1:', nbig, 'of which set-openers', bigop, '| top changed tokens:', [(asc(w),c) for w,c in big.most_common(12)])
R=pd.DataFrame(rows_ls, columns=['k','isop','iswait','p_orig','p_E6','pW_orig','pW_E6'])
print(R.groupby('k')[['p_orig','p_E6','pW_orig','pW_E6']].mean().round(3).T, '\n n per k', R.groupby('k').size().tolist())
```