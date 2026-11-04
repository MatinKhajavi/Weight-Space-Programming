The attention check shows that 19.4's strongly negative output (-5.58) is poorly aimed. Only 0.11–0.12 of its attention lands on earlier copies of the upcoming token, about twice uniform. Head 19.6 is better aimed (0.28 in loops, 0.24 on normal text) with a moderately negative output (-2.09). The biggest remaining copier is 23.4: it puts 0.56 of its attention on such copies in loops (0.40 on normal text) and has the highest copy score (5.93). It was part of edit B, which hurt. Attention times score is only a rough proxy, so I will measure each head's actual contribution to the logit of the token produced next. The contribution comes from the recorded attention weights and residuals of the C windows, passed through the head's value path, query-dependent output gate and output projection, then read out at the final layer. This is a linear decomposition of recorded activations, not a rerun of the network. Comparing in-loop and pre-loop positions shows which heads still push the repeat inside loops and whether 19.6/19.4 actively oppose it. A suppressor with sizeable negative in-loop contribution and small effects on normal text would justify an amplification test.

```python
# Hypothesis: per-head direct contributions (recorded attention x value path x gate x o_proj, read out at final layer) show which heads push / oppose the copied token in C loops vs normal text
def head_dla(l_, b_):
    L_=m18['lengths'][b_]; s_=np.array(ids_all[b_])
    Hl=torch.tensor(np.asarray(np.load(f'/work/fwd_18/hidden_{l_}.npy', mmap_mode='r')[b_,:L_],dtype=np.float32))
    lnw_=1+load_original(f'model.language_model.layers.{l_}.input_layernorm.weight')
    X_=Hl/Hl.pow(2).mean(-1,keepdim=True).sqrt()*lnw_
    Wq_=load_original(f'model.language_model.layers.{l_}.self_attn.q_proj.weight'); Wv_=load_original(f'model.language_model.layers.{l_}.self_attn.v_proj.weight'); Wo_=load(f'model.language_model.layers.{l_}.self_attn.o_proj.weight')
    A__=torch.tensor(np.asarray(np.load(f'/work/fwd_18/attn_{l_}.npy', mmap_mode='r')[b_,:, :L_, :L_],dtype=np.float32))
    qf=(X_@Wq_.T).view(L_,8,512); h24b=torch.tensor(np.asarray(h24[b_,:L_],dtype=np.float32)); r24=h24b.pow(2).mean(-1).sqrt()
    nxt=torch.tensor(np.r_[s_[1:], s_[-1]])
    Unext=(E_[nxt]-mu_E)*g_                       # [T,1024]
    out={}
    for h_ in range(8):
        gk=h_//4; v_=X_@Wv_[gk*256:(gk+1)*256].T
        ho=torch.sigmoid(qf[:,h_,256:])*(A__[h_]@v_)
        o_=ho@Wo_[:, h_*256:(h_+1)*256].T
        out[h_]=((o_*Unext).sum(-1)/r24).numpy()
    return out
dla_tab={}
for b_ in range(2):
    L_=m18['lengths'][b_]; s_=ids_all[b_]; P_=find_period(s_)
    t0=L_-1
    while t0-1-P_>=0 and s_[t0-1]==s_[t0-1-P_]: t0-=1
    th=s_.index(248068)
    inl=np.arange(t0+2*P_, L_-1); pre=np.arange(th+50, t0-1)
    for l_ in [15,19,23]:
        o_=head_dla(l_,b_)
        for h_ in range(8): dla_tab.setdefault(f'{l_}.{h_}',[]).append((o_[h_][inl].mean(), o_[h_][pre].mean()))
dt=pd.DataFrame({k:{'in':np.mean([x[0] for x in v]),'pre':np.mean([x[1] for x in v])} for k,v in dla_tab.items()}).T
dt['diff']=dt['in']-dt['pre']
print(dt.sort_values('in').round(3).to_string())
```