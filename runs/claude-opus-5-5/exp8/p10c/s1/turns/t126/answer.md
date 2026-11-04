On the 512 selection problems, v4 against the original is +48.4 points (standard error 2.2; 250 wrong-to-right, 2 right-to-wrong). The gain is the same in both sources: GSM8K +50.8 (13% to 64%) and MATH +47.7 (7% to 55%). That gives a concrete prediction for the held-out check: a paired gain of roughly 40-50 points in both sources. The fresh set has 256 problems that appear in none of the 768 ids used so far: 128 MATH stratified by MATH-500 level and 128 GSM8K. I will now write the pass-9 result and the pass-10 plan into the notes and print the ids for the final forward block. The current /work/edited is exactly v4, which is what the final pass will test and what will be submitted.

```python
# Hypothesis: (bookkeeping) record pass-9 necessity result + pass-10 plan, then print fresh ids for the final 'both' pass on v4.
with open('/work/notes.md', 'a') as f:
    f.write("""
## Pass 9 RESULT (gen_14, v7 = v4 with L11H1/L15H7 restored, same 512): acc 53.5 vs v4 57.2: paired -3.71 SE 1.86 (36 w->r, 55 r->w); setA -3.9, diagS -3.5, gsm -3.1, math -3.9. Mechanism as predicted: cut 140->176 (38 of 55 losses are cut in v7); commonly finished acc unchanged (.841 v4 vs .837 v7). -> early copy-head halving is necessary; KEEP v4.
- v4 vs original on the 512: +48.4 SE 2.2 (250 w->r, 2 r->w); gsm 13.3->64.1 (+50.8 SE 4.4), math 7.3->54.9 (+47.7 SE 2.6).
- v4 restored from /work/best_4 and verified (only the intended column blocks differ from original). FINAL CANDIDATE = v4.
## Pass 10 plan: 'both' (v4 vs original) on /work/confirm_ids.json = 256 never-used problems (128 MATH stratified 11/23/27/33/34 by level + 128 GSM8K, seed 777), n=1, 32k. Prediction: paired gain ~ +40-50 in both sources.
""")
print(json.dumps(fresh))
```