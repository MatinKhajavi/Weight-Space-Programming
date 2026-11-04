The run failed because I had reused the variable name `d` for the paired-difference array in the previous run, so the fwd_18 directory path was lost. The notes update before the failing line did go through. I will restore the path and rerun the head selection, the control edit and the S1 ID printout, without appending the notes again. The plan is unchanged. The control is the Wait brake ×2 plus three low-induction heads in layers 15 and 23 zeroed, in place of 15.7/23.0/23.5. If the effect is specific to the copying heads, this control should score like brake-only (44.9% on S1), not like C (60.2%).

```python
# (retry) Specificity control: pick lowest-induction heads in L15/L23 from C loop windows; control edit = brake x2 + those heads zeroed (copy heads restored)
d='/work/fwd_18'
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
print(edited(), 'brake still x2:', bool(torch.allclose(load('model.language_model.layers.22.mlp.down_proj.weight')[:,573],2*W22o[:,573])))
grp=collections.OrderedDict()
for i in S1: pre,num=i.rsplit('_',1); grp.setdefault(pre,[]).append(num)
for pre,nums in grp.items(): print(pre, len(nums), ' '.join(nums))
```