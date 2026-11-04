The final norm multiplies by (1+w): that version reproduces the top-k logit gaps to within 0.03, versus 0.88 for plain w. I can now do an exact direct-logit attribution. For each layer I'll split the residual update into its MLP part (saved activations times down-projection) and its mixer part (attention or DeltaNet, the remainder). I'll project each onto the correctly normalised unembedding of the actual next token, and compare loop with pre-loop positions in probe 2.

```python
# Hypothesis: the loop excess is in the attention (mixer) part of layers 15/19/23, not their MLPs
wF=1+gF; sc=np.sqrt((H[24]**2).mean(-1)+1e-6)
U2=Eemb[ids2[1:]] ; pos_l=lp_pos[lp_pos<len(ids2)-1]; pos_p=pp_pos
rowsD=[]
for j in range(24):
    Wd=load(f'model.language_model.layers.{j}.mlp.down_proj.weight').numpy()
    act=np.load(f'/work/fwd_4/mlp_act_{j}.npy',mmap_mode='r')[1]
    res_=[]
    for pos in (pos_p,pos_l):
        mo=np.asarray(act[pos],dtype=np.float32)@Wd.T
        mx=H[j+1][pos]-H[j][pos]-mo
        f_=lambda v: float(((v/sc[pos,None])*wF*U2[pos]).sum(-1).mean())
        res_+= [f_(mx), f_(mo)]
    rowsD.append((j, *[round(x,2) for x in res_]))
print('layer: mixer_pre mlp_pre mixer_loop mlp_loop')
for r_ in rowsD: print(r_, '(FA)' if r_[0] in (3,7,11,15,19,23) else '', ' dMix',round(r_[3]-r_[1],2),' dMLP',round(r_[4]-r_[2],2))
```