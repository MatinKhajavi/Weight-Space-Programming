The paired analysis supports keeping the early copy-head edits. Undoing them (v7) costs 3.71 points (standard error 1.86; 36 problems wrong-to-right against 55 right-to-wrong). The loss is similar in both halves of the set (setA -3.9, diagS -3.5) and in both sources (GSM8K -3.1, MATH -3.9). It works through the predicted mechanism: cut-off answers rise from 140 to 176, and 38 of the 55 losses are cut in v7. Accuracy on problems both versions finish is unchanged (0.841 against 0.837), so halving L11H1/L15H7 does not damage the reasoning itself; it only cuts down the self-copying loops. So v4 stays the final candidate. Now I restore it from /work/best_4 and check every edited tensor against the original. The check should show only the intended column-level changes: L16 neuron 998 zeroed, the eight copy heads at ×0.5, and L22 neuron 573 at ×4, with nothing else differing.

```python
# Hypothesis: restoring best_4 reproduces exactly v4: L16 col998=0, L11H1/L15H7/L19H5,H7/L23H0,H4,H5,H7 o_proj x0.5, L22 col573 x4; nothing else.
for f in sorted(os.listdir('/work/best_4')):
    name = f[:-4]; t = torch.from_numpy(np.load('/work/best_4/' + f)).float()
    save_edit(name, t)
print(edited())
for name in edited():
    e = load(name); o = load_original(name)
    assert torch.equal(e, torch.from_numpy(np.load('/work/best_4/' + name + '.npy')).float())
    dc = torch.nonzero((e != o).any(0)).flatten().tolist(); dr = torch.nonzero((e != o).any(1)).flatten().tolist()
    # summarize changed column blocks and ratios
    blocks = []; 
    for c in dc:
        if blocks and c == blocks[-1][1] + 1: blocks[-1][1] = c
        else: blocks.append([c, c])
    rat = []
    for a, b in blocks:
        oo = o[:, a:b+1]; ee = e[:, a:b+1]; m = oo.abs() > 1e-8
        rat.append('%d:%d x%.3f' % (a, b+1, (ee[m] / oo[m]).mean().item()))
    print(name.replace('model.language_model.', ''), tuple(e.shape), 'changed rows', len(dr), '| col blocks', rat)
```