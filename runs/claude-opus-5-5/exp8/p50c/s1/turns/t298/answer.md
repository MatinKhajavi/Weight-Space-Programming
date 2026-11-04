The specificity control came out clearly. With the Wait brake ×2 plus three non-copying heads zeroed (15.6, 23.6, 23.3) instead of the copying heads, accuracy on S1 is 27.3% with 70.7% of answers cut. That is not just short of C's 60.2%: it is worse than brake-only (44.9%, cut 53.9%). So C's gain does not come from ablating any three heads in layers 15 and 23. It is specific to 15.7, 23.0 and 23.5, the heads that copy earlier line starts. The control heads even carry function the model needs to finish. I will now compute the paired numbers for the record and restore C exactly. I will also set up the matching control for the brake: edit A plus doubling the other built-in layer-22 brake, 22.387 (the Maybe brake), instead of 22.573. If the Wait-specific target matters, this should score like A (35.2% on S1), not like C.

```python
# Record specificity control (paired vs brake-only gen_15 and C gen_13 on S1), restore C exactly, then set up brake-target control = A + 22.387 x2 (instead of 22.573)
CT=smap(['/work/gen_30.jsonl']); BR=smap(['/work/gen_15.jsonl']); C13=smap(['/work/gen_13.jsonl'])
def pdiff(X,Y):
    ks=[k for k in X if k in Y]; dd=np.array([(X[k][0]=='ok')-(Y[k][0]=='ok') for k in ks],float)
    return len(ks), round(dd.mean()*100,1), round(dd.std(ddof=1)/np.sqrt(len(dd))*100,1)
print('ctrl vs brake-only', pdiff(CT,BR), ' ctrl vs C', pdiff(CT,C13), ' brake-only vs C', pdiff(BR,C13))
for l,hs in ((15,[7]),(23,[0,5])):
    Wo=load_original(f'model.language_model.layers.{l}.self_attn.o_proj.weight').clone()
    for h in hs: Wo[:,h*256:(h+1)*256]=0
    save_edit(f'model.language_model.layers.{l}.self_attn.o_proj.weight',Wo)
print(os.listdir('/work/best_C_x2'))
with open('/work/notes.md','a') as f:
    f.write("- PASS 24 = gen_30: SPECIFICITY CONTROL (brake x2 + non-copy heads 15.6/23.6/23.3 zeroed, induction .02/.00/.02 vs 15.7 .20, 23.5 .13) on S1: acc 27.3% cut 70.7% vs brake-only 44.9% / C 60.2%: ctrl vs brake-only %s, vs C %s (n, diff pts, SE). => C gain specific to copy heads 15.7/23.0/23.5; random heads hurt. Restored C.\n" % (pdiff(CT,BR), pdiff(CT,C13)))
# brake-target control: A heads + 22.387 x2 (22.573 original)
W22=W22o.clone(); W22[:,387]=2*W22o[:,387]; save_edit('model.language_model.layers.22.mlp.down_proj.weight',W22)
print(edited(), '573 orig:', bool(torch.allclose(load('model.language_model.layers.22.mlp.down_proj.weight')[:,573],W22o[:,573])), '387 x2:', bool(torch.allclose(load('model.language_model.layers.22.mlp.down_proj.weight')[:,387],2*W22o[:,387])))
```