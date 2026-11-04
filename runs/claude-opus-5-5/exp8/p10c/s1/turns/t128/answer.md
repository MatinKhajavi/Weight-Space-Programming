The held-out check confirms v4. On 256 problems never used before, accuracy went from 12.9% to 63.3%: a paired gain of +50.4 points (standard error 3.2), with 131 problems going wrong-to-right and only 2 right-to-wrong. Both sources improve: GSM8K from 15.6% to 67.2% and MATH from 10.2% to 59.4%. The cut-off share fell from 89% to 24% and the mean length halved (29.6k to 14.5k tokens). This sits at the top of the predicted +40 to +50 range, so the dose choices did not overfit the 512 selection problems. I will submit v4 unchanged. One last bookkeeping run computes per-source and per-level paired statistics and accuracy on the problems both models finish (a check that the reasoning itself is not damaged). It also verifies that /work/edited is byte-identical to the verified v4 copy and records the result in the notes.

```python
# Final bookkeeping: per-source/level paired stats on the held-out confirmation, check no damage on commonly finished, verify /work/edited == v4, update notes.
o15 = table(readj('/work/gen_15_original.jsonl')); e15 = table(readj('/work/gen_15_edited.jsonl'))
cid = [i for i in fresh if i in o15 and i in e15]
co_ = np.array([o15[i]['c'] for i in cid], float); ce_ = np.array([e15[i]['c'] for i in cid], float)
lvl = {r['id']: str(r.get('level', '')).strip()[-1:] for r in math}
rows = [('all', np.ones(len(cid), bool)), ('gsm', np.array([o15[i]['src']=='gsm8k' for i in cid])), ('math', np.array([o15[i]['src']=='math' for i in cid]))]
rows += [('L%s' % L, np.array([lvl.get(i) == L for i in cid])) for L in '12345']
out = []
for nm, m in rows:
    dd = ce_[m] - co_[m]
    out.append('%-4s n %3d orig %.3f v4 %.3f diff %+.1f SE %.1f w->r %d r->w %d' % (nm, m.sum(), co_[m].mean(), ce_[m].mean(), 100*dd.mean(), 100*dd.std(ddof=1)/np.sqrt(m.sum()), (dd>0).sum(), (dd<0).sum()))
print('\n'.join(out))
cuto = np.array([o15[i]['cut'] for i in cid]); cute = np.array([e15[i]['cut'] for i in cid])
bf = (~cuto) & (~cute)
print('cut orig %d v4 %d | commonly finished %d: acc orig %.3f v4 %.3f | v4 finished %d acc %.3f' % (cuto.sum(), cute.sum(), bf.sum(), co_[bf].mean(), ce_[bf].mean(), (~cute).sum(), ce_[~cute].mean()))
for name in edited():
    assert torch.equal(load(name), torch.from_numpy(np.load('/work/best_4/' + name + '.npy')).float())
print('edited == best_4 (v4):', sorted(n.replace('model.language_model.', '') for n in edited()))
with open('/work/notes.md', 'a') as f:
    f.write("\n## Pass 10 RESULT (gen_15, both, 256 never-used problems): v4 vs original 12.9 -> 63.3, paired +50.4 SE 3.2 (131 w->r, 2 r->w); gsm 15.6 -> 67.2, math 10.2 -> 59.4; cut 88.7%% -> 24.2%%; mean tok 29.6k -> 14.5k. Prediction (+40-50) confirmed out of sample.\n" + '\n'.join('- ' + o for o in out) + "\n- FINAL SUBMISSION = v4 (/work/edited == /work/best_4).\n")
```