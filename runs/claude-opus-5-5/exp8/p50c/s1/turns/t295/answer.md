X6 is a clean negative result. The doubled Maybe brake worked mechanically: line-initial "Maybe" fell from 2.38 to 0.05 per 1k tokens and lowercase "maybe" from 0.55 to 0.04, the same threshold-like collapse as Wait. But accuracy did not improve (-2.0 points, SE 1.9; GSM8K exactly 0.0, MATH -3.9 ± 2.6), and cut answers were unchanged. So the Maybe-doubt spirals are a symptom of being stuck, not a cause. The doubt reappears in other words ("Is it", "Possib..." rose slightly). Unlike Wait lines, Maybe lines do not drive copy-loops, and on MATH the "Maybe" exploration seems mildly useful. C is restored. With this, the routes I can see for further accuracy gains (loop brakes, extra heads, doubt brakes) are exhausted. Remaining failures look capability-limited: correct answers are short (median 6.6k tokens), the long ones are lost. I will use the spare budget to strengthen the causal evidence for C with a specificity control: brake ×2 with three non-copying heads zeroed in the same layers (15 and 23), instead of the three Wait-copying heads. If C's gain comes from those specific heads, this control should score like brake-only (44.9% on S1), not like C (60.2%). This run picks the control heads by their measured induction attention in C's loop windows, builds the control edit, and prints the S1 IDs.

```python
# Specificity control: pick non-copying heads in L15/L23 (lowest induction attention in C loop windows), build control edit = brake x2 + those 3 heads zeroed
with open('/work/notes.md','a') as f:
    f.write("\n## Session 5\n- PASS 23 = gen_29: X6 = C + 22.387 down col x2 (built-in maybe-family brake: col z Maybe 18, maybe 20, Perhaps 17, Possibly 18; act one-signed, -5 before Maybe vs -0.5 other line starts). SH512 vs C: acc 58.8 vs 60.7, diff -2.0 SE 1.9 (gsm 0.0, math -3.9 SE 2.6); LI Maybe 2.38->0.05/1k, maybe .55->.04; cut unchanged. => Maybe spirals are a SYMPTOM (doubt re-routes; Maybe exploration mildly useful in MATH). Reverted to C.\n- Repetition-tracking analysis (fwd_18): copied-token DLA grows only +0.7 c1->c3-5 (mostly L23 mixer); cycle0 already ~19 logits -> loops are predictable fixed points, no single reinforcer.\n- Whole-head ablation cross-tabs: X1 (23.4) shifts cut->wrong; ok->wrong ~30-36/512 in all = churn level.\n")
ind={}
for l in (3,7,11,15,19,23):
    Aat=np.load(f'{d}/attn_{l}.npy',mmap_mode='r')
    for hh in range(8):
        v=[]
        for w,(p,on,K) in enumerate(info):
            t=np.arange(on,Ls[w]-1); v.append(float(np.asarray(Aat[w,hh,t,t+1-p],dtype=np.float32).mean()))
        ind[(l,hh)]=np.mean(v)
print({f'{l}.{hh}':round(v,2) for (l,hh),v in ind.items() if l in (15,23)})
ctrl15=min([h for h in range(8) if h!=7], key=lambda h: ind[(15,h)])
ctrl23=sorted([h for h in range(8) if h not in (0,2,4,5,7)], key=lambda h: ind[(23,h)])[:2]
print('control heads: 15.%d'%ctrl15, ['23.%d'%h for h in ctrl23])
for l,hs in ((15,[ctrl15]),(23,ctrl23)):
    Wo=load_original(f'model.language_model.layers.{l}.self_attn.o_proj.weight').clone()
    for h in hs: Wo[:,h*256:(h+1)*256]=0
    save_edit(f'model.language_model.layers.{l}.self_attn.o_proj.weight',Wo)
print(edited())
grp=collections.OrderedDict()
for i in S1: pre,num=i.rsplit('_',1); grp.setdefault(pre,[]).append(num)
for pre,nums in grp.items(): print(pre, len(nums), ' '.join(nums))
```