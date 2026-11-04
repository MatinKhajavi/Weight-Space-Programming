The pass-2 data are as expected: two 1,950-token natural loop texts (original model) with float16 hidden states for all 25 residual positions and MLP activations for all 24 layers. Now I run the decomposition.

For each layer, the MLP output comes from the saved activations times the down-projection. The attention or DeltaNet output is the residual difference minus that MLP output. Each output is read out onto the centred unembedding of the token that actually comes next, scaled by the final norm, separately for the pre-loop thinking and for positions inside the loop after its first full cycle.

The per-block contributions should sum to the total target logit, which I check. If a few full-attention layers carry most of the in-loop increase, the copy engine is the full-attention heads and stronger head edits are justified. If DeltaNet layers dominate, head scaling cannot stop lock-in and I need a different target.

```python
# Hypothesis: in-loop push onto the copied next token comes from specific blocks; per-block direct logit attribution, pre-loop vs in-loop (fwd_4, original)
think_id = enc('<think>')[0]; full = {3,7,11,15,19,23}
Eo_mean = Eo.mean(0)
res_blocks = {}
for i in range(2):
    ids = I4[i].tolist(); T = len(ids)
    on, p = onset_period(ids); st = ids.index(think_id) + 1 if think_id in ids else 0
    tgt = np.array(ids[1:] + [ids[-1]])
    h24 = torch.tensor(np.asarray(Hs[24][i], np.float32))
    rms = torch.sqrt((h24**2).mean(1, keepdim=True) + 1e-6)
    G = (Eo[tgt] - Eo_mean) * wf / rms                                  # (T,1024)
    pre = np.arange(st, on - 1); loop = np.arange(on + p, T - 1)
    rows = []
    tot = (h24 * G).sum(1).numpy()
    for l in range(24):
        A_ = torch.tensor(np.asarray(np.load(f'/work/fwd_4/mlp_act_{l}.npy', mmap_mode='r')[i], np.float32))
        mo = A_ @ Wd[l].T
        ao = torch.tensor(np.asarray(Hs[l+1][i], np.float32)) - torch.tensor(np.asarray(Hs[l][i], np.float32)) - mo
        da = (ao * G).sum(1).numpy(); dm = (mo * G).sum(1).numpy()
        rows.append((l, da[pre].mean(), da[loop].mean(), dm[pre].mean(), dm[loop].mean()))
    h0 = (torch.tensor(np.asarray(Hs[0][i], np.float32)) * G).sum(1).numpy()
    res_blocks[i] = rows
    print(f"text {i}: onset {on} period {p} think_start {st} | total target logit (centered) pre {tot[pre].mean():.2f} loop {tot[loop].mean():.2f} | emb pre {h0[pre].mean():.2f} loop {h0[loop].mean():.2f}")
    print('  layer type  attn_pre attn_loop  d_attn | mlp_pre mlp_loop d_mlp')
    for (l, ap, al, mp, ml) in rows:
        print(f"  L{l:2d} {'FULL' if l in full else 'dnet'}  {ap:+6.2f} {al:+6.2f} {al-ap:+6.2f} | {mp:+6.2f} {ml:+6.2f} {ml-mp:+6.2f}")
```