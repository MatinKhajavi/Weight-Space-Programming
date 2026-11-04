Silencing heads 23.5 and 23.6 has a clear, dose-like effect on loop stickiness:
- The chance that the sampler leaves the loop in a period rises from 0.069 to 0.187 at the 4th repetition and from 0.006 to 0.068 at the 5th (no presence penalty). With the presence penalty it rises from 0.18 to 0.39 and from 0.09 to 0.23.
- The average loop-token probability falls only modestly, about 0.01–0.02.

(The k=6 row is distorted because the texts end mid-period.) So these heads are a causal part of the lock-in, but two heads alone remove only part of the repetition boost. Next I measure how strongly each part of layer-23 attention pushes the actual loop token: the two silenced heads (recovered exactly as the difference between passes) versus the remaining six, and how each push grows with repetition. If the other heads (23.0/4/7 induction, 23.2 same-token) carry the rest of the +2.5-logit repetition boost, they are the next candidates. If not, the remaining boost comes from earlier layers (e.g., 15.7).

```python
# Decompose L23 attention's direct push on the loop token: removed heads (5,6) vs remaining heads, by repetition k
nmW = lambda l_: f'model.language_model.layers.{l_}.mlp.down_proj.weight'
Wd23 = load_original(nmW(23)).numpy()
fn_w = load_original('model.language_model.norm.weight').numpy()
print('final norm weight mean', fn_w.mean().round(3), '| gF mean', float(gF.mean()), '-> zero-centred' if abs(fn_w.mean())<0.5 else '')
ma7 = np.load(f'{F7}/mlp_act_23.npy', mmap_mode='r'); h24e = np.load(f'{F7}/hidden_24.npy', mmap_mode='r')
WUn = WU.numpy() if hasattr(WU,'numpy') else WU
gFn = gF.numpy() if hasattr(gF,'numpy') else gF
rows=[]
for b_ in range(14):
    n_ = L4[b_]; rep_ = np.asarray(REP[b_])
    H23o = HS[23][b_,:n_]; H24o = HS[24][b_,:n_]
    att_o = H24o - H23o - MA[23][b_,:n_] @ Wd23.T
    att_e = h24e[b_,:n_] - h23e[b_,:n_] - ma7[b_,:n_] @ Wd23.T
    o56 = att_o - att_e
    rms = np.sqrt((H24o**2).mean(-1, keepdims=True) + 1e-6)
    nxt = ids4[b_,1:n_]; U = WUn[nxt] - WUc_mean[None,:]
    d56 = ((o56[:-1]*gFn)/rms[:-1] * U).sum(-1); drest = ((att_e[:-1]*gFn)/rms[:-1] * U).sum(-1)
    for t_ in range(n_-1):
        rows.append(dict(k=rep_[t_+1], d56=d56[t_], drest=drest[t_], dlp=lp7[b_,t_]-lp4[b_,t_]))
Dd = pd.DataFrame(rows)
print(Dd[Dd.k>0].groupby('k')[['d56','drest','dlp']].mean().round(3).to_string())
print('lead (k=0, pre-loop completion+prompt) mean:', Dd[Dd.k==0][['d56','drest','dlp']].mean().round(3).to_dict())
```