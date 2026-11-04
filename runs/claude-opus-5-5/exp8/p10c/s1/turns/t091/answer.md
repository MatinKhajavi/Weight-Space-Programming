Text 1 shows the same pattern as text 0, so the copy engine replicates across both loops:

- **Layer-level push onto the copied token (text 1):** layer 23 attention +5.0, layer 15 attention +3.0, layer 23 MLP +1.1, layer 19 attention +0.6 and MLP +0.7. The DeltaNet layers stay near zero or negative.
- **Heads by in-loop attention to the matching position of the previous cycle:** layer 15 head 7 (0.75/0.66 in texts 0/1) and layer 11 head 1 (0.53/0.33), both already halved. Then layer 23 heads 5, 0, 4, 7 (0.25–0.54), and layer 19 heads 5, 7, 6 (up to 0.55 in text 1).

So the largest remaining direct copy writer is layer 23's attention, whose heads v2 never touched.

There is a risk. Layer 23 attention also contributes heavily to ordinary predictions before any loop (+8.7 of a 27.7 target logit). Blindly halving its heads could damage normal copying of numbers and names.

To choose heads that are loop specialists, I will profile what each candidate head does before the loop starts versus inside it:
- attention mass on the first (sink) token;
- mass on the problem statement;
- mass on the last 16 tokens;
- a fair induction mass, summed over all earlier occurrences of the current token.

A head that mostly idles on the sink before the loop but copies the previous cycle inside it is a safe, targeted edit. A head that already copies heavily before the loop would be risky to cut.

```python
# Hypothesis: some late copy heads (L19/L23) are loop specialists (idle/sink before the loop, copy inside it) and are safer to suppress than general copy heads
cands = ['L11H1','L15H7','L15H5','L19H5','L19H6','L19H7','L23H0','L23H4','L23H5','L23H7','L23H1','L19H2']
prof = collections.defaultdict(dict)
for i in range(2):
    ids = I4[i].tolist(); T = len(ids); on, p = tail_loop(ids); st = ids.index(think_id) + 1
    pre_q = np.arange(st + 64, on, 2); loop_q = np.arange(on + 2 * p, T, 2)
    occ = collections.defaultdict(list)
    for s in range(1, T): occ[ids[s - 1]].append(s)          # positions s whose previous token is ids[s-1]
    def ind_mass(Ah, q):
        ks = [s for s in occ[ids[q]] if s < q - 1]
        return Ah[q, ks].sum() if ks else 0.0
    for name in cands:
        L, hd = int(name[1:name.index('H')]), int(name[name.index('H') + 1:])
        Ah = np.asarray(np.load(f'/work/fwd_4/attn_{L}.npy', mmap_mode='r')[i, hd], np.float32)
        for tag, Q in [('pre', pre_q), ('loop', loop_q)]:
            sink = Ah[Q, 0].mean(); prm = Ah[Q, 1:st].sum(1).mean()
            rec = np.mean([Ah[q, max(0, q - 16):q + 1].sum() for q in Q]); ind = np.mean([ind_mass(Ah, q) for q in Q])
            prof[name].setdefault(tag, []).append((sink, prm, rec, ind))
print('head   | PRE: sink prompt recent16 induction | LOOP: sink prompt recent16 induction   (avg of 2 texts)')
for name in cands:
    pr = np.mean(prof[name]['pre'], 0); lo = np.mean(prof[name]['loop'], 0)
    print(f"{name:6s} | {pr[0]:.2f} {pr[1]:.2f} {pr[2]:.2f} {pr[3]:.2f} | {lo[0]:.2f} {lo[1]:.2f} {lo[2]:.2f} {lo[3]:.2f}")
```