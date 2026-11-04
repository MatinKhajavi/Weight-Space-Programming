E8 is installed. It changes L23's o_proj by 0.47 relative (vs 0.17 for E6), mostly from silencing two of the eight heads. Only about 9% of those heads' output lay in the opener subspace, and that part is kept in its reversed form. I'll record the OV findings and the E8 plan in the notes. Then I'll fetch the D-set ids so pass 23 runs E8 on exactly the problems and seeds of gen_16 (E6 on D, n=2). That gives a paired E8 − E6 comparison.

The prediction: if the general copy push of L23 h5/h6 sustains E6's leftover loops, E8 should lower the cut share below E6's 48% and raise accuracy. It should not lower accuracy on samples both versions finish. If cut share and accuracy don't move, the remaining loops are carried by other copy paths, and I keep E6.

```python
# Log OV analysis + E8 plan in notes; print D ids (gen_16 order) for the paired E8-vs-E6 generation pass
with open('/work/notes.md','a') as f:
    f.write("""
## Restart-4
- Verified all gens used max_tokens 32768 (eval protocol). E6 cut-offs pooled D/H/H2: 496 cuts = 318 exact loops (median period 57 chars) + 178 near-loops (near_rep median .97: wrap-up cycles, bullet restatements, a few counters).
- E1 (L23 h5,h6 zeroed, orig) finished-acc .984 (60/61), both-finished 19/19 -> no reasoning damage.
- OV copy score (weights only, embedding inputs, z of self-logit): L11 h4 +8.0, h7 +8.5 (copiers), h6 -9.8, h0 -5.8 (copy-suppressors); L15 h1 +5.4; L23 h2 -6.3 (suppressor); L23 h5/h6 ~0 (embedding probe not valid at L23; in-context ablation DLA showed h5+6 push loop token +4.4->+5.7).
- E8 = E6 + L23 heads 5,6: out-of-opener-subspace output removed (keep reversed in-subspace part). rel change L23 .47 (vs E6 .17). Saved /work/cand_E8. PASS 23: E8 on D n=2, paired vs gen_16 (E6).
""")
sets=json.load(open('/work/sets.json')); print(type(sets), list(sets.keys()) if isinstance(sets,dict) else len(sets))
Dids=[r['id'] for r in E6rows['D']]
print(len(Dids)); print(json.dumps(Dids))
```