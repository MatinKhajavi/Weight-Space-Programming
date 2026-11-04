The cut texts under C are dominated by verbatim line stutters. A single line is repeated thousands of times ("Jacob catches 161.", "Diego *buys* apples.", "$\triangle PR = 65^\circ$.", "Faces = 2×2 + 3×3 = 13"), or a short block of bullet lines cycles, as in "Kept [Mike] = 23". Many of these sit at deep bullet indentation (32–80 spaces), and one GSM8K text shows a long "Maybe the tickets were…" brainstorming spiral before looping. The "0" answers are explicit give-ups ("I will output 0."), the end of confused reasoning. Line stutters are pure copying of the previous line. Since 15.7, 23.5 and 23.0 are already zeroed, some other copying component must carry them. The C windows in fwd_18 include full attention maps for all six full-attention layers. I will compute each head's induction mass inside the loop: attention to the token that follows the previous occurrence of the current token, i.e. the token about to be copied. I will compare it with the same score on normal pre-loop text. Heads with high in-loop and low pre-loop induction mass would be candidates for the remaining copy route. If no head stands out, the copying is carried by the linear-attention layers or spread too thinly to ablate.

```python
# Hypothesis: under C, a few remaining full-attention heads carry the in-loop copying (attend to token after previous occurrence); compute per-head induction mass in-loop vs pre-loop on fwd_18
res_ind={}
for l_ in [3,7,11,15,19,23]:
    A__=np.load(f'/work/fwd_18/attn_{l_}.npy', mmap_mode='r')
    if l_==3: print('attn shape', A__.shape, A__.dtype)
    for b_ in range(2):
        L_=m18['lengths'][b_]; s_=ids_all[b_]; P_=find_period(s_)
        t0=L_-1
        while t0-1-P_>=0 and s_[t0-1]==s_[t0-1-P_]: t0-=1
        th=s_.index(248068) if 248068 in s_ else 0
        Ab=np.asarray(A__[b_,:, :L_, :L_], dtype=np.float32)      # [H, T, T]
        inl=np.arange(t0+2*P_, L_-1); pre=np.arange(th+50, t0-1)
        # in-loop: mass on all previous-cycle positions t-kP+1
        m_in=np.zeros(Ab.shape[0])
        for t_ in inl:
            js=np.arange(t_-P_+1, t0-1, -P_); js=js[js>=0]
            m_in+=Ab[:, t_, js].sum(-1)
        m_in/=len(inl)
        # pre-loop generic induction: positions j with ids[j-1]==ids[t], j<=t
        sarr=np.array(s_); m_pre=np.zeros(Ab.shape[0]); cnt_=0
        for t_ in pre[::3]:
            js=np.nonzero(sarr[:t_]==sarr[t_])[0]+1; js=js[js<=t_]
            if len(js)==0: continue
            m_pre+=Ab[:, t_, js].sum(-1); cnt_+=1
        m_pre/=max(cnt_,1)
        for h_ in range(Ab.shape[0]): res_ind.setdefault((l_,h_),[]).append((m_in[h_], m_pre[h_]))
tab_ind=sorted([(k, np.mean([x[0] for x in v]), np.mean([x[1] for x in v])) for k,v in res_ind.items()], key=lambda x:-x[1])
print('head (layer.head): in-loop mass, pre-loop induction mass  [C windows]')
print([ (f'{k[0]}.{k[1]}', round(a,2), round(b,2)) for k,a,b in tab_ind[:16]])
```