X1 (C plus head 23.4 zeroed) reached 60.7% on the 512 items, the same as C's 60.7% (311/512) on the same items and seeds. The composition changed: cut-offs fell from 127 to 99 (24.8% → 19.3%), while wrong or unfinished answers rose from about 75 to 102. So 23.4 does carry part of the remaining loop copying, but removing it converts loops into wrong answers rather than correct ones. That fits it also supporting correct computation, as edit B suggested. I will compute the exact paired transitions (wrong→right versus right→wrong) by source, and check whether the newly wrong answers are short (computation damage) or long (failed searches that merely stopped looping). That decides whether a partial dose could keep the loop reduction without the damage.

```python
# Paired comparison X1 (gen_22) vs C (gen_13 S1 + gen_16_edited H1): transitions, by source; nature of new wrongs
g22 = load_gen('/work/gen_22.jsonl')
gC = {**{i: g13[i] for i in S1}, **{i: g16e[i] for i in H1}}
def st(s): return 'ok' if str(s['correct']) == 'True' else ('cut' if str(s['cut']) == 'True' else 'wrong')
def paired(gx, gy, ids, label):
    cx = np.array([st(gx[i]) == 'ok' for i in ids]); cy = np.array([st(gy[i]) == 'ok' for i in ids])
    d = cy.astype(int) - cx.astype(int); se = d.std(ddof=1) / np.sqrt(len(d))
    print(f'{label}: base {cx.mean():.3f} new {cy.mean():.3f} diff {100 * d.mean():+.1f} SE {100 * se:.1f} | W->R {(d == 1).sum()} R->W {(d == -1).sum()}')
paired(gC, g22, ids512, 'X1 vs C all'); paired(gC, g22, [i for i in ids512 if i.startswith('gsm')], ' gsm'); paired(gC, g22, [i for i in ids512 if not i.startswith('gsm')], ' math')
trans = Counter((st(gC[i]), st(g22[i])) for i in ids512); print(dict(trans))
nw_ = [int(g22[i]['tokens']) for i in ids512 if st(gC[i]) == 'ok' and st(g22[i]) == 'wrong']
nr_ = [int(g22[i]['tokens']) for i in ids512 if st(gC[i]) != 'ok' and st(g22[i]) == 'ok']
print('new wrong (from ok) tokens median', np.median(nw_) if nw_ else None, '| new right tokens median', np.median(nr_) if nr_ else None)
cw = [int(g22[i]['tokens']) for i in ids512 if st(gC[i]) == 'cut' and st(g22[i]) == 'wrong']; print('cut->wrong n', len(cw), 'tokens median', np.median(cw) if cw else None)
```