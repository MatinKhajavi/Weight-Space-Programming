The run failed before computing anything: `tok` is a raw `tokenizers` Tokenizer, so `encode` returns an Encoding object and I need its `.ids`. The analysis is unchanged. I decompose the logit difference between " Wait" and the main alternative (" So" in window 0, " Okay" in window 1) at every loop cycle start, by layer, into mixer and MLP parts. Large positive layer-19 and layer-23 mixer contributions at later cycles would mean copying re-selects "Wait". Consistent late-MLP contributions already at the first cycle would mean a doubt feature.

```python
# Hypothesis: at loop cycle starts, ' Wait' beats the alternative via copy (L19/L23 mixers) rather than a fixed MLP 'doubt' feature
WAIT = 13428
def tid(s): return tok.encode(s).ids[0]
def dla_diff(b, tq, y_tok, alt_tok):
    h24 = np.asarray(H11[24][b, tq]).astype(np.float64); rms = np.sqrt((h24**2).mean(1)+1e-6)
    u = (Eemb[y_tok]-Eemb[alt_tok]) * (1.0+wn.numpy())
    rows=[]
    for l in range(1,25):
        dl = np.asarray(H11[l][b,tq]) - np.asarray(H11[l-1][b,tq])
        act = np.asarray(np.load(f'/work/fwd_11/mlp_act_{l-1}.npy', mmap_mode='r')[b, tq])
        Wd_ = load(pre+f'{l-1}.mlp.down_proj.weight').numpy()
        mo = act @ Wd_.T; mix = dl - mo
        rows.append(((mix@u)/rms, (mo@u)/rms))
    emb = (np.asarray(H11[0][b,tq])@u)/rms
    return emb, rows
for b, alt in ((0, tid(' So')), (1, tid(' Okay'))):
    x = ids11[b,:lens11[b]]; Pc=per[b]; c0 = 922 if b==0 else 761
    starts = np.array([c0 + k*Pc for k in range(0, 40) if c0+k*Pc < lens11[b]])
    assert all(x[starts]==WAIT), (b, x[starts][:5])
    tq = starts - 1
    emb, rows = dla_diff(b, tq, WAIT, alt)
    tot = emb + sum(r[0]+r[1] for r in rows)
    print('win', b, 'n starts', len(starts), 'alt', alt, 'total DLA diff mean %.2f (first %.2f, later mean %.2f)' % (tot.mean(), tot[0], tot[2:].mean()))
    top = sorted([(abs(r[0][2:].mean()), l, 'mix', r[0][0], r[0][2:].mean()) for l, r in enumerate(rows)] + [(abs(r[1][2:].mean()), l, 'mlp', r[1][0], r[1][2:].mean()) for l, r in enumerate(rows)], reverse=True)[:10]
    for z in top: print('   L%-2d %s  first-cycle %+.2f  later %+.2f' % (z[1], z[2], z[3], z[4]))
```