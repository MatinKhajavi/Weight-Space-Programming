X4 (C plus head 23.7 zeroed) is worse than C: 57.2% vs 60.7%, with cut-offs up from 24.8% to 28.7%. Like 11.1, head 23.7 is load-bearing and its removal increases looping. Every candidate copying head has now been tested on its own:
- 15.7, 23.0 and 23.5 help;
- 23.4 is neutral (it trades loops for errors);
- 11.1 and 23.7 hurt;
- 19.5 has no effect on loop probability.

So C's head set is final, and I revert 23.7. The brake is saturated, and the remaining failures are diverse and mostly reflect genuine difficulty, so C is my final edit.

The most valuable use of the next passes is an independent replication of C vs the original on problems never touched. I will build H2: 128 MATH problems level-stratified like MATH-500 (11/23/27/33/34 for levels 1–5) plus 128 GSM problems, all from the unused held-out pools. A "both" pass on them gives a seed-paired difference. This should reproduce the H1 result (+49 ± 3 points) within error.

```python
# Revert X4 (23.7) -> C; build fresh held-out H2 (never used) for a paired C-vs-original replication
m26 = acc_map('/work/gen_26.jsonl'); d = np.array([m26[k][0]-mC[k][0] for k in keys], float)
print('X4 vs C: %.1f SE %.1f, W->R %d R->W %d' % (100*d.mean(), 100*d.std(ddof=1)/np.sqrt(len(d)), (d>0).sum(), (d<0).sum()))
W23 = load('model.language_model.layers.23.self_attn.o_proj.weight').clone(); W23[:, 7*256:8*256] = W0[:, 7*256:8*256]
save_edit('model.language_model.layers.23.self_attn.o_proj.weight', W23)
print('L23', [round(float(load('model.language_model.layers.23.self_attn.o_proj.weight')[:,h*256:(h+1)*256].norm()/W0[:,h*256:(h+1)*256].norm()),2) for h in range(8)], edited())
used2 = used | set(O_ids)
rng2 = np.random.default_rng(23)
hm = {L: [i for i in pools['H_math'] if i not in used2 and mt[i]['level']==f'Level {L}'] for L in range(1,6)}
H2 = sum([list(rng2.choice(hm[L], c, replace=False)) for L,c in zip(range(1,6),(11,23,27,33,34))], []) + list(rng2.choice([i for i in pools['H_gsm'] if i not in used2], 128, replace=False))
json.dump(H2, open('/work/H2_ids.json','w'))
with open('/work/notes.md','a') as f: f.write("- PASS 21 = gen_26 (X4 = C + 23.7 zeroed, SH512): acc 57.2 vs C 60.7 (diff %.1f SE %.1f), cut 28.7 -> 23.7 load-bearing; reverted to C. Head set final: 15.7/23.0/23.5.\n- H2 (/work/H2_ids.json): fresh held-out 128 MATH (MATH-500 level mix) + 128 GSM. Pass 22 (planned): both, H2, C.\n" % (100*d.mean(), 100*d.std(ddof=1)/np.sqrt(len(d))))
print(json.dumps(H2))
```