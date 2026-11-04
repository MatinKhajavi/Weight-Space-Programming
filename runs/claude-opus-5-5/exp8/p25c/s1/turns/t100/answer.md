The DLA explains why So survived E4b. The two edited layers now work against So, and increasingly so with repetition: L15 attention −1.4 (k=0) to −3.5 (k=5), L19 −0.6 to −1.3. The new engine is L23 attention, which pushes So by +1.2 at k=0 and +4.1 at k=5. Its growth (+2.5) accounts for essentially all of the rising margin (−0.86 → +1.52). This fits the original diagnosis that L23's induction heads drive verbatim loop continuation; once L15/L19 stopped copying openers, L23's copy took over at the decision points. The L23 MLP partly opposes it (−0.5 → −2.5). The natural next edit (E5) applies the same centred opener-subspace reversal to L23 o_proj. Before building it I check two things. First, which tokens compete with So at these points: openers inside the 5-token set, or outsiders such as But/This/`</think>`? Second, the first-order predicted change of the So margin for β=1,2,3, for the centred 5-opener subspace and for an uncentred variant.

```python
# Alternatives at So decision points + first-order predicted effect of reversing opener subspace in L23 o_proj
alt_ctr = collections.Counter(); alt_mass = collections.Counter()
for (i, t, k, sid) in dec:
    for a, lv in zip(tki13[i, t][:6], tkl13[i, t][:6]):
        a = int(a)
        if a in SO_IDS: continue
        alt_ctr[a] += 1; alt_mass[a] += float(np.exp(lv))
show = [(a, repr(tok.decode([a])), alt_ctr[a], round(alt_mass[a]/len(dec),3)) for a, _ in alt_mass.most_common(25)]
print([s for s in show if s[1].isascii()][:22])
# Subspaces (readout space, gF-weighted unembedding), orthonormal bases
def basis(vecs):
    Qb, _ = np.linalg.qr(np.stack(vecs, 1)); return Qb
R5 = np.stack([gFn_*WUn_[i] for i in oid])          # Wait, So, Let, Okay, If (bare)
B_c5 = basis(list(R5 - R5.mean(0)))[:, :4]
B_u5 = basis(list(R5))
pred = collections.defaultdict(list)
for (i, t, k, sid) in dec:
    x24 = np.asarray(H13[24][i, t]); rms24 = np.sqrt((x24**2).mean() + 1e-6)
    alt = [(int(a), float(np.exp(lv))) for a, lv in zip(tki13[i, t], tkl13[i, t]) if int(a) not in SO_IDS]
    ws = sum(w for _, w in alt); ualt = sum(w*WUn_[a] for a, w in alt)/ws
    u = gFn_*(WUn_[sid]-ualt)/rms24
    a23v = (np.asarray(H13[24][i, t]) - np.asarray(H13[23][i, t])) - Wd_all[23] @ np.asarray(M13[23][i, t])
    for nm, Bq in (('c5', B_c5), ('u5', B_u5)):
        pa = Bq @ (Bq.T @ a23v)
        pred[(nm, min(k,5))].append(float(pa @ u))
for nm in ('c5','u5'):
    print(nm, 'a23 projected So-margin contribution by k:', {k: round(float(np.mean(pred[(nm,k)])),2) for k in range(6)},
          '-> beta=3 first-order change = -3x')
```