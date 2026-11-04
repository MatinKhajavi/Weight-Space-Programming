Using the model's own residuals, the readout now behaves sensibly. Known loop drivers come out as copiers (23.5 at 4.66, 15.7 at 4.32), along with 23.4, 15.6, 19.2, 15.5 and 23.2. 23.0 scores weakly and 11.1 near zero, consistent with 11.1 feeding later heads rather than copying directly. One head stands out on the other side: 19.4 scores -5.58, so whatever token it attends to gets pushed down, the signature of a copy-suppression head. Then come 15.4 (-2.21) and 19.6 (-2.09). A suppressor only helps against loops if it actually attends to earlier occurrences of the token about to be produced. So for every head in layers 11–23 I will measure attention mass on all earlier positions holding the actual next token, inside loops and on normal pre-loop text. If 19.4 puts substantial mass there, especially inside loops, it is a natural anti-repetition head, and amplifying it is a principled, testable edit. If its attention goes elsewhere, its negative score is irrelevant to loops.

```python
# Hypothesis: head 19.4 is a copy-suppression head: it attends to earlier occurrences of the upcoming token (NTO mass), in loops and on normal text
res_nto={}
for l_ in [11,15,19,23]:
    A__=np.load(f'/work/fwd_18/attn_{l_}.npy', mmap_mode='r')
    for b_ in range(2):
        L_=m18['lengths'][b_]; s_=np.array(ids_all[b_]); P_=find_period(ids_all[b_])
        t0=L_-1
        while t0-1-P_>=0 and s_[t0-1]==s_[t0-1-P_]: t0-=1
        th=int(np.nonzero(s_==248068)[0][0]) if (s_==248068).any() else 0
        Ab=np.asarray(A__[b_,:, :L_, :L_], dtype=np.float32)
        for reg,pos_ in [('in',np.arange(t0+2*P_, L_-1)),('pre',np.arange(th+50,t0-1))]:
            acc_=np.zeros(8); cnt_=0
            for t_ in pos_:
                js=np.nonzero(s_[:t_+1]==s_[t_+1])[0]          # earlier positions holding the upcoming token
                if len(js)==0: continue
                acc_+=Ab[:,t_,js].sum(-1); cnt_+=1
            for h_ in range(8): res_nto.setdefault((f'{l_}.{h_}',reg),[]).append(acc_[h_]/max(cnt_,1))
nto_tab=pd.DataFrame({reg:{k[0]:np.mean(v) for k,v in res_nto.items() if k[1]==reg} for reg in ['in','pre']})
nto_tab['ctx_self']=[dict(ctx_scores).get(k,np.nan) for k in nto_tab.index]
print(nto_tab.sort_values('pre',ascending=False).head(14).round(2).to_string())
print(nto_tab.loc[['19.4','15.4','19.6','15.3','19.3']].round(2).to_string())
```