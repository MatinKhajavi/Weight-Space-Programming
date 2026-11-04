The helper returns the position of the last indentation token; the bullet "    *   Wait" tokenizes as [262, 348, 256, 13428]. So the line's first word sits at offset +1, and the returned position is itself the decision point. There are 27 doubt-word decisions across the two texts, 15 of them " Wait", against about 190 other line starts.

I will rerun the per-block attribution onto the " Wait" logit at the correct decision positions. Doubt-next decisions are compared with other line starts, with neuron 998 shown separately as a reference: earlier work put it at about +1.3. If the late full-attention layers (15/19/23) push Wait substantially more at doubt decisions, the copy heads also sustain doubt, which favours suppressing them. If the remaining push sits in MLPs, the head edit should stay modest.

```python
# Hypothesis: besides L16 N998, the Wait push at line-start decisions comes from late full-attention copy layers (L15/L19/L23); decision position = returned position s, next word = ids[s+1]
rowsW = collections.defaultdict(list)
for i in range(2):
    ids = I4[i].tolist(); T = len(ids); st = ids.index(think_id) + 1
    S = np.array([s for s in line_start_positions(ids) if st < s < T - 1])
    isD = np.array([ids[s + 1] in D_ids for s in S]); tpos = S
    h24 = torch.tensor(np.asarray(Hs[24][i], np.float32))[tpos]; rms = torch.sqrt((h24**2).mean(1, keepdim=True) + 1e-6)
    Gw = ((Eo[wait_id] - Eo_mean) * wf)[None] / rms
    tot = (h24 * Gw).sum(1).numpy(); rowsW['total'].append((tot[isD].mean(), tot[~isD].mean()))
    for l in range(24):
        A_ = torch.tensor(np.asarray(np.load(f'/work/fwd_4/mlp_act_{l}.npy', mmap_mode='r')[i][tpos], np.float32)); mo = A_ @ Wd[l].T
        ao = torch.tensor(np.asarray(Hs[l+1][i][tpos], np.float32)) - torch.tensor(np.asarray(Hs[l][i][tpos], np.float32)) - mo
        da = (ao * Gw).sum(1).numpy(); dm = (mo * Gw).sum(1).numpy()
        rowsW[f'L{l}a'].append((da[isD].mean(), da[~isD].mean())); rowsW[f'L{l}m'].append((dm[isD].mean(), dm[~isD].mean()))
        if l == 16:
            n998 = (A_[:, 998:999] * Wd[16][:, 998][None] * Gw).sum(1).numpy(); rowsW['N998'].append((n998[isD].mean(), n998[~isD].mean()))
diffs = {k: np.mean([a - b for a, b in v]) for k, v in rowsW.items()}
lev = {k: np.mean([a for a, b in v]) for k, v in rowsW.items()}
print('total Wait-logit: at doubt decisions', round(lev['total'], 2), 'diff vs other line starts', round(diffs['total'], 2), '| N998 diff', round(diffs['N998'], 2))
print('top + blocks:', ', '.join(f"{k}:{v:+.2f}" for k, v in sorted(diffs.items(), key=lambda kv: -kv[1])[:12] if k != 'total'))
print('top - blocks:', ', '.join(f"{k}:{v:+.2f}" for k, v in sorted(diffs.items(), key=lambda kv: kv[1])[:8]))
print('full-attn layer diffs:', {k: round(float(diffs[k]), 2) for k in ['L3a','L7a','L11a','L15a','L19a','L23a']})
print('sum attn all layers', round(sum(diffs[f'L{l}a'] for l in range(24)), 2), '| sum mlp all layers', round(sum(diffs[f'L{l}m'] for l in range(24)), 2))
```