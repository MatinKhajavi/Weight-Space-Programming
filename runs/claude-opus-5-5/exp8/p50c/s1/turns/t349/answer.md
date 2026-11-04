Even on AIME problems, far beyond this model's level, the edit removes some looping. The original model hit the token limit on all 256 attempts and scored 0%; C cut that to 79% and solved 3.5% (paired +3.5 ± 1.4, 9 wrong→right, 0 right→wrong). On hard problems capability is the binding limit, but the repair still turns loops into finished attempts. Ten fresh MATH/GSM sets gave +45.7 to +51.6 points each.

I will use five of the remaining seven passes to replicate the brake decomposition on fresh problems. On SH512, BW matched C exactly (paired 0.0 ± 1.9). BW adds only the brake column's Wait-logit component (verified with the corrected norm gain: 0.310 vs 0.310 for the full column). Running BW on H6–H10, where C's results already exist on identical seeds, gives 1,280 paired items, about ±1.3 points. The prediction is that BW ≈ C on fresh data, confirming that what the brake repairs is "Wait" suppression. This run rebuilds BW and logs H8–H10 and AIME.

```python
# R16 hypothesis: BW (Wait-logit component only) equals C on fresh problems too; rebuild BW for 5 fresh-set paired passes vs C's existing results (H6-H10). Log H8-H10 and AIME.
d22=load_original(P+'22.mlp.down_proj.weight').clone(); d22[:,573]=col573+colW
for k in [P+'15.self_attn.o_proj.weight', P+'23.self_attn.o_proj.weight']: save_edit(k, C_TARGET[k])
save_edit(P+'22.mlp.down_proj.weight', d22)
chk=float((load(P+'22.mlp.down_proj.weight')[:,573]-(col573+colW)).abs().max())
same15=float((load(P+'15.self_attn.o_proj.weight')-C_TARGET[P+'15.self_attn.o_proj.weight']).abs().max()); same23=float((load(P+'23.self_attn.o_proj.weight')-C_TARGET[P+'23.self_attn.o_proj.weight']).abs().max())
print('BW rebuilt:', [n.replace(P,'L') for n in edited()], 'col check', chk, 'L15/L23 equal C:', same15, same23)
with open('/work/notes.md','a') as fh:
    fh.write("- PASS 40 = gen_48 (both, H8): orig 11.7 (cut 90.2) | C 61.7 (gsm 67.2, math 56.3, cut 21.9); diff +50.0 SE 3.1, W->R 128, R->W 0.\n- PASS 41 = gen_49 (both, H9): orig 14.8 (cut 85.9) | C 60.9 (gsm 66.4, math 55.5, cut 21.5); diff +46.1 SE 3.2, W->R 120, R->W 2.\n- PASS 42 = gen_50 (both, H10): orig 8.2 (cut 92.2) | C 54.3 (gsm 63.3, math 45.3, cut 26.2); diff +46.1 SE 3.2, W->R 119, R->W 1. Held-out H1-H10 (2560 paired): +45.7..+51.6 each.\n- PASS 43 = gen_51 (both, AIME128 x2): orig 0.0 (cut 100) | C 3.5 (cut 79.3); diff +3.5 SE 1.4, W->R 9, R->W 0 (capability-limited, loops reduced).\n- BW rebuilt for fresh replication: passes 44-48 = BW on H6..H10 (edited), pair with C's gen_43/44/48/49/50 edited. MUST restore C afterwards (C_TARGET / /work/best_final).\n")
```