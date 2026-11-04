The per-head contributions close the attention route. The suppressor heads (19.4, 15.3, 23.3, 19.6, 15.4) lower the actual next token by about the same amount inside loops as on normal text. They are general calibration heads, and amplifying them would act like a non-specific flattening. Among the promoters, 23.6, 23.2 and 23.4 contribute heavily to the correct next token on normal text too (1.7–2.8 logits). Only 23.4 has a sizeable loop-specific excess (+0.66), and it is too important for normal prediction to remove, consistent with edit B's harm. So the remaining copying runs through general-purpose prediction heads, and no further head edit is justified. I will screen MLP neurons in layers 12–23 the same way: each neuron's direct contribution to the actual next token (activation × down column, read out at the final layer), inside loops versus normal text, on both C windows. A neuron with a consistently large loop-specific contribution would be either a repetition amplifier to damp or a native anti-repetition neuron to strengthen. Its effect must be consistent across both windows to count.

```python
# Hypothesis: a few MLP neurons (layers 12-23) contribute loop-specifically to the copied token (amplifiers) or against it (native anti-repetition), consistently across both C windows
regs={}
for b_ in range(2):
    L_=m18['lengths'][b_]; s_=ids_all[b_]; P_=find_period(s_)
    t0=L_-1
    while t0-1-P_>=0 and s_[t0-1]==s_[t0-1-P_]: t0-=1
    th=s_.index(248068); regs[b_]=(np.arange(t0+2*P_, L_-1), np.arange(th+50, t0-1))
neu_rows=[]
for l_ in range(12,24):
    Wd_l=load(f'model.language_model.layers.{l_}.mlp.down_proj.weight')       # current weights (C)
    Aml=np.load(f'/work/fwd_18/mlp_act_{l_}.npy', mmap_mode='r')
    per_b=[]
    for b_ in range(2):
        L_=m18['lengths'][b_]; s_=np.array(ids_all[b_]); inl,pre=regs[b_]
        r24=torch.tensor(np.sqrt((np.asarray(h24[b_,:L_],dtype=np.float32)**2).mean(-1)))
        nxt=torch.tensor(np.r_[s_[1:], s_[-1]])
        Un=((E_[nxt]-mu_E)*g_)/r24[:,None]                                     # [T,1024]
        proj=(Un@Wd_l)                                                          # [T,3584]: logit per unit act of each neuron on next token
        act=torch.tensor(np.asarray(Aml[b_,:L_],dtype=np.float32))
        con=(act*proj).numpy()                                                  # direct contribution per neuron per position
        per_b.append((con[inl].mean(0), con[pre].mean(0)))
    din=np.stack([pb[0]-pb[1] for pb in per_b])                                  # [2,3584] loop-specific excess
    for n_ in range(3584):
        neu_rows.append((l_, n_, din[0,n_], din[1,n_], per_b[0][0][n_], per_b[1][0][n_], per_b[0][1][n_], per_b[1][1][n_]))
nr=pd.DataFrame(neu_rows, columns=['l','n','ex0','ex1','in0','in1','pre0','pre1'])
nr['ex_min']=np.where(np.sign(nr.ex0)==np.sign(nr.ex1), np.sign(nr.ex0)*np.minimum(nr.ex0.abs(), nr.ex1.abs()), 0)
print('loop-specific promoters (consistent):'); print(nr.sort_values('ex_min',ascending=False).head(10).round(3).to_string(index=False))
print('loop-specific opposers (consistent):'); print(nr.sort_values('ex_min').head(10).round(3).to_string(index=False))
print('total loop excess from MLPs 12-23 per window:', nr.ex0.sum().round(2), nr.ex1.sum().round(2))
```