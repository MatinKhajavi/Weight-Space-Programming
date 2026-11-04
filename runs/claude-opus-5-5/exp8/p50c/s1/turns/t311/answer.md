Strengthening the duplicate-token head 23.6 did not help. C + 23.6 ×1.5 scored 59.1% against C's 60.7% on the same seeds, with slightly more cut answers (27.6% vs 24.8%). GSM8K and MATH both dipped a little. The summary shows 511 instead of 512 generations, so one ID was probably mistyped or dropped in my request; the paired comparison is unaffected. So 23.6's weak suppression is already well calibrated, and the specificity control's damage must come from the other two heads (23.3 attends to the problem statement, 15.6 to recent context) or from their combination. I will restore C and record the result. In the same run I will tabulate the paired per-benchmark differences of every C variant tested on SH512 (23.4, 11.1 and 23.7 ablations, the Maybe brake, 23.6 ×1.5). This checks whether any of them shows a consistent benefit on MATH, whose remaining failures are mostly cut-offs (32% cut vs 12.5% wrong on H4), that a gentler version could exploit.

```python
# Restore C; record X7; per-benchmark paired diffs of all C-variants on SH512 (look for any consistent MATH benefit)
for l,hs in ((15,[7]),(23,[0,5])):
    Wc=load_original(f'model.language_model.layers.{l}.self_attn.o_proj.weight').clone()
    for h in hs: Wc[:,h*256:(h+1)*256]=0
    save_edit(f'model.language_model.layers.{l}.self_attn.o_proj.weight',Wc)
print('C restored:', all(torch.equal(list(load_file('/work/best_C_x2/'+fn).values())[0].float(), load('model.language_model.'+fn.replace('.safetensors',''))) for fn in os.listdir('/work/best_C_x2')))
res_={}
for nm,p in [('X1 23.4=0','/work/gen_22.jsonl'),('X3 11.1=0','/work/gen_25.jsonl'),('X4 23.7=0','/work/gen_26.jsonl'),('X6 Maybe-brake x2','/work/gen_29.jsonl'),('X7 23.6 x1.5','/work/gen_34.jsonl')]:
    X=smap([p]); out=[]
    for src in ('gsm8k','math'):
        ks=[k for k in X if k in C and C[k][1]==src]; dd=np.array([(X[k][0]=='ok')-(C[k][0]=='ok') for k in ks],float)
        dc=np.array([(X[k][0]=='cut')-(C[k][0]=='cut') for k in ks],float)
        out.append(f'{src} n{len(ks)} acc {dd.mean()*100:+.1f}+-{dd.std(ddof=1)/np.sqrt(len(dd))*100:.1f} cut {dc.mean()*100:+.1f}')
    res_[nm]=out; print(nm,' | '.join(out))
with open('/work/notes.md','a') as f:
    f.write("- Head attention stats (C loop windows): 23.6 = duplicate-token head (dup .93), OV mildly suppressive (own-token z -0.6); 23.2 strongly suppressive (z -5) but self .64 (anti-stutter; risky for repeated digits); 19.0-19.3, 23.3 attend to prompt (.2-.3).\n- PASS 28 = gen_34: X7 = C + 23.6 x1.5 on SH512 (511 gens): acc 59.1 vs C 60.7, cut 27.6 vs 24.8 => no gain; reverted to C.\n- MATH cut texts (145, distinctive refs): ref before onset .39 vs control .22, boxed ref 0 => commit failure small; MATH cuts mostly never-solved.\n")
```