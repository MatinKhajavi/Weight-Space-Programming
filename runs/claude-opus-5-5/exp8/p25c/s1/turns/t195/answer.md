The attribution confirms the mechanism.

At loop decision points from the third repetition on (58 positions), the original's Wait-over-alternatives margin is +13.1 logits. It is pushed by attention in L19 (+3.2), L15 (+2.1) and L23 (+1.8), plus the L23 MLP (+4.4), against a constant suppression from the L22 MLP (−4.0). Under E6 the three attention terms become −6.6, −4.2 and −0.65 (a combined swing of about −18.5 logits). Partial compensation follows: the L22 MLP relaxes its suppression (+4.0), the L23 MLP adds +1.4 and the L15 MLP +1.3. The net margin falls to +1.8, so the repeated opener now loses to "Let"/"So"/"If".

In normal reasoning before the loop (46 positions), the net change is only −0.6. There L19 attention pushes Wait by about 0 logits, so reversing its opener component has nothing to flip. The opener is chosen mainly by MLPs, which E6 doesn't touch. This is why the edit is selective: it flips the copying contribution that grows with repetition and leaves context-driven choices alone.

For the last pass I'll do a third independent paired confirmation (original vs E6, same seeds) on 256 problems never used in any pass. They will be 160 MATH problems stratified by level like D, and 96 GSM8K. That gives one more test of the final model under the exact evaluation protocol. First I'll record these findings and draw the fresh set, excluding every id used before.

```python
# Notes + draw fresh held-out set H3 (160 MATH stratified like D + 96 GSM8K), excluding every id used in any previous pass
with open('/work/notes.md','a') as f:
    f.write("""
- PASS 24 (fwd_25) TF E6 on 14 compact loop texts (ids identical to fwd_4). Per-period escape prob (T.6/top-p.95) k3/4/5: orig .29/.07/.006, E2 .31/.08/.016, E3 .71/.71/.75, E6 .85/.88/.91.
  Decision points (Wait positions) P(Wait) by k1..6: orig .57/.72/.84/.91/.96/.98; E3 .23/.26/.26/.20/.16/.13; E6 .23/.25/.11/.04/.02/.01 (repetition now LOWERS reuse). Lead k0: .32 -> .28.
  E6 alternatives at Wait positions: Let, So, If, Okay. Lead tokens dlogp mean -.018, median 0, |d|>1 3 pct (25 tokens, 21 = set openers).
  DLA Wait-vs{Let,So,If,Okay} margin k>=3: orig +13.1 (L19 attn +3.2, L15 +2.1, L23 +1.8, L23 mlp +4.4, L22 mlp -4.0) -> E6 +1.8 (L19 -6.6, L15 -4.2, L23 -0.65; compensation L22 mlp -4.0->0.0, L23 mlp +1.4, L15 mlp +1.3).
  Lead k0 margin 7.98 -> 7.38: L19 attn pushes ~0 in normal reasoning -> reversal selective to repetition-driven copying.
- PASS 25 plan: 'both' E6 vs orig on fresh H3 (third held-out replication).
""")
used=set()
for fn in os.listdir('/work'):
    if fn.startswith('gen_') and fn.endswith('.jsonl'):
        for l in open('/work/'+fn): used.add(json.loads(l)['id'])
used |= {it['id'] for it in tc}
sh=json.load(open('/work/sets_H.json')); print('sets_H keys', list(sh.keys()) if isinstance(sh,dict) else type(sh))
for v in (sh.values() if isinstance(sh,dict) else []): used |= set(v if isinstance(v,list) else [])
used |= set(Dids)
mrowsA=[json.loads(l) for l in open('/practice/math_train.jsonl')]; growsA=[json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]
rng3=np.random.default_rng(31337); quota={1:14,2:29,3:34,4:41,5:42}; H3=[]
for lv,q in quota.items():
    pool_=[r['id'] for r in mrowsA if int(r.get('level',0))==lv and r['id'] not in used]; H3 += list(rng3.choice(pool_, q, replace=False))
gp=[r['id'] for r in growsA if r['id'] not in used]; H3 += list(rng3.choice(gp, 96, replace=False))
print('used', len(used), 'H3', len(H3), 'overlap', len(set(H3)&used), 'unique', len(set(H3)))
sh['H3']=H3; json.dump(sh, open('/work/sets_H.json','w'))
print(json.dumps([str(x) for x in H3]))
```