The ablation run finished. The mean token logprob over the windows dropped only slightly (W1 -0.464 → -0.486, W2 -0.256 → -0.308), but the decision points are what matter. I'll compare fwd_19 (C + 19.5 ablated) with fwd_18 (C) on three things. First, the probability of the copied first word on each repeated line in W1. Second, the mean logprob of loop tokens in W2 and in the late W1 loop. Third, the change in the layer-19 mixer's logit attribution (the exact direct effect of 19.5) and in layer 21's attribution (a downstream effect).

```python
# Hypothesis: ablating 19.5 lowers p(copy) at loop decision points, mostly on early repeats; measure direct+downstream
D2='/work/fwd_19/'
lp2=np.load(D2+'token_logprobs.npy'); ids2=np.load(D2+'input_ids.npy'); assert (ids2==ids).all()
Hs2={l:np.load(D2+f'hidden_{l}.npy',mmap_mode='r') for l in range(25)}
Ma2={l:np.load(D2+f'mlp_act_{l}.npy',mmap_mode='r') for l in range(24)}
Wd_cur2={l:load(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy() for l in [19,20,21,22,23]}
def mixer_dla(Hs,Ma,Wdd,b,p,y,alt,l):
    h24=Hs[24][b,p-1].astype(np.float32); rms=np.sqrt((h24**2).mean()+1e-6)
    d=(Wemb[y]-Wemb[alt]).numpy()*gfin/rms
    hd=Hs[l+1][b,p-1].astype(np.float32)-Hs[l][b,p-1].astype(np.float32)
    mo=Wdd[l]@Ma[l][b,p-1].astype(np.float32)
    return float((hd-mo)@d), float(mo@d)
print('line  p_copy(C) -> p_copy(C-19.5) | L19mix C->abl | L21mix C->abl')
for p,y,alt,mix,mlp,pr in rowsD:
    if p<666: continue
    pa=float(np.exp(lp2[0,p-1]))
    m19a,_=mixer_dla(Hs2,Ma2,Wd_cur2,0,p,y,alt,19); m21a,_=mixer_dla(Hs2,Ma2,Wd_cur2,0,p,y,alt,21)
    if p<800 or p%5==0:
        print(p, repr(tstr(y)), round(pr,2),'->',round(pa,2),'|',round(mix[19],1),'->',round(m19a,1),'|',round(mix[21],1),'->',round(m21a,1))
inW2=range(W2['n_head']+150,lens[1])
print('W2 loop mean logprob C',round(float(np.mean([lp[1,p-1] for p in inW2])),4),'abl',round(float(np.mean([lp2[1,p-1] for p in inW2])),4))
inW1=range(800,lens[0])
print('W1 late-loop mean logprob C',round(float(np.mean([lp[0,p-1] for p in inW1])),4),'abl',round(float(np.mean([lp2[0,p-1] for p in inW1])),4))
pre1=range(W1['n_head']+20,660)
print('W1 pre-loop mean logprob C',round(float(np.mean([lp[0,p-1] for p in pre1])),4),'abl',round(float(np.mean([lp2[0,p-1] for p in pre1])),4))
```