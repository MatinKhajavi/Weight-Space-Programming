The paired result for E6-strong against E6 on D is +1.2 ± 2.7 overall. By source it is GSM8K +8.9 ± 4.7, with the cut share falling from 31% to 22% and both-finished accuracy rising from 0.80 to 0.85, and MATH −3.4 ± 3.1, with the cut share unchanged at 58–59% and both-finished accuracy falling from 0.89 to 0.82 (n = 72). One reading is that stronger reversal helps short grade-school reasoning but starts to disturb longer competition-style derivations; neither source difference is conclusive yet. Since the final score covers both MATH-500 and GSM8K, this needs more paired data than D can give. I will run E6-strong on the held-out set H, where E6 and the original already exist (gen_18), plus a fresh set H2 with the same composition. A later "both" pass with E6 installed on H2 will then pair against these E6-strong results. That design yields about 1,000 paired E6-strong-vs-E6 samples (D + H + H2) and a second held-out E6-vs-original estimate, for two passes. Here I build H2, excluding every problem in D and H.

```python
# Build fresh held-out set H2 (160 MATH stratified like MATH-500 + 96 GSM8K), disjoint from D and H; print ids H+H2 for the E6-strong pass.
excl=set(used_ids)|set(H)
rngH2=np.random.default_rng(4242); H2=[]
for L_,q_ in quotaH.items():
    pool=sorted(o['id'] for o in mrows if o.get('level')==L_ and o['id'] not in excl); H2+=list(rngH2.choice(pool,q_,replace=False))
gpool=sorted(o['id'] for o in grows if o['id'] not in excl); H2+=list(rngH2.choice(gpool,96,replace=False))
H2=[str(x) for x in H2]; print(len(H2), 'overlap with D/H:', len(set(H2)&excl))
json.dump({'H':H,'H2':H2}, open('/work/sets_H.json','w'))
print(json.dumps(H2, separators=(',',':')))
```