Head 19.0 retrieves earlier line openings in general, not doubt markers specifically. In edit A's windows, its attention from line starts went to earlier 'Wait' (0.56) and 'Okay' (0.33) tokens. In C's windows, with 'Wait' gone, it goes to whatever openers exist ('Only', 'Remaining', 'But', 'maybe'). Head 19.5 behaves similarly ('Wait' 0.60 under A; 'She', '$', '*', 'If' under C). So layer 19 copies line-opening style, which is what made 'Wait' self-reinforcing before the brake. Unlike the induction heads, 19.0 matches by word type rather than by previous token, so removing it might not damage number copying. Before spending a pass, I will check from recorded activations how strongly layer 19's attention output pushes the actual next line-opening word in C's windows, inside versus before loops, compared with layer 23 and the MLPs. Only a large push would make a 19.0 test worthwhile.

```python
# DLA at line starts in C windows: how much do L19/L23 attention outputs (and MLPs) push the actual next line-opening token? in-loop vs pre-loop
d = '/work/fwd_18'; m_ = json.load(open(d + '/meta.json')); ii = np.load(d + '/input_ids.npy')
H = {L: np.load(f'{d}/hidden_{L}.npy', mmap_mode='r') for L in [14, 15, 18, 19, 22, 23, 24]}
MA = {L: np.load(f'{d}/mlp_act_{L}.npy', mmap_mode='r') for L in [15, 19, 22, 23]}
Wd = {L: load(f'model.language_model.layers.{L}.mlp.down_proj.weight').numpy() for L in [15, 19, 22, 23]}
R18 = regions_for(d)
for b in range(2):
    n, on_, ids = R18[b]; pos = ls_positions(ids[:n]); pos = pos[pos > 90]
    h24 = np.asarray(H[24][b, :n], np.float32); rms = np.sqrt((h24 ** 2).mean(-1))
    U = (E[torch.tensor(ids[pos + 1].astype(np.int64))] * fnw).numpy()          # unembedding of actual next token
    def comp(vec): return (vec[pos] * U).sum(-1) / rms[pos]
    out_ = {}
    for L in [15, 19, 22, 23]:
        mlp = np.asarray(MA[L][b, :n], np.float32) @ Wd[L].T
        mixer = np.asarray(H[L][b, :n], np.float32) - np.asarray(H[L - 1][b, :n], np.float32) - mlp
        out_[f'mix{L}'] = comp(mixer); out_[f'mlp{L}'] = comp(mlp)
    inl = pos >= on_
    print('win', b, 'line starts pre', int((~inl).sum()), 'loop', int(inl.sum()), '|', {k: (round(float(v[~inl].mean()), 2), round(float(v[inl].mean()), 2)) for k, v in out_.items()})
```