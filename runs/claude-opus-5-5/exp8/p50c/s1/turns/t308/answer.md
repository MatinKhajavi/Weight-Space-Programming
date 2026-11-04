Failure to commit is not a large category on MATH. Of 145 MATH cut-offs with distinctive reference answers, the correct answer appears before the loop in 39%, against a 22% chance rate for other problems' answers in the same texts. A boxed correct answer appears in none. So at most about one in six MATH cut-offs had really reached the answer; the rest never solved the problem. One result does deserve a follow-up: the specificity control. Zeroing the "non-copying" heads 15.6, 23.6 and 23.3 pushed cut answers from 54% to 71% (-17.6 points vs brake-only). So these heads actively help the model make progress and finish, perhaps by keeping it anchored to the problem statement or by suppressing repetition. If their attention shows a clear anti-loop role (for example strong attention to the prompt, or to earlier copies of the current token), strengthening them would be a principled test, with ablation's -17.6 as the other end of the dose curve. This run tabulates attention statistics for all heads in layers 11, 15, 19 and 23 in C's loop windows. I will also prepare the test edit (C with these three heads ×1.5), to run only if the patterns support it.

```python
# What do the 'needed' non-copy heads 15.6/23.6/23.3 attend to (prompt anchoring? duplicate tokens?) vs others; prepare C + those heads x1.5
TH=248068
stats={}
for l in (11,15,19,23):
    Aat=np.load(f'{d}/attn_{l}.npy',mmap_mode='r')
    for hh in range(8):
        acc_=collections.defaultdict(list)
        for w,(p,on,K) in enumerate(info):
            x=ids[w,:Ls[w]]; tp=int(np.where(x==TH)[0][0]); t=np.arange(on,Ls[w]-1)
            M_=np.asarray(Aat[w,hh,on:Ls[w]-1,:Ls[w]],dtype=np.float32)
            acc_['bos'].append(M_[:,0].mean()); acc_['prompt'].append(M_[:,1:tp+1].sum(1).mean())
            acc_['prev'].append(M_[np.arange(len(t)),t-1].mean()); acc_['self'].append(M_[np.arange(len(t)),t].mean())
            rec=np.array([M_[i,max(0,tt-64):tt].sum() for i,tt in enumerate(t)]); acc_['rec64'].append(rec.mean())
            dup=np.array([M_[i,:tt][x[:tt]==x[tt]].sum() for i,tt in enumerate(t)]); acc_['dup'].append(dup.mean())
        stats[(l,hh)]={k:round(float(np.mean(v)),2) for k,v in acc_.items()}
print('head: bos prompt prev self rec64 dup | ind')
for (l,hh),s in stats.items():
    tag=' <CTRL' if (l,hh) in ((15,6),(23,6),(23,3)) else (' <ZEROED' if (l,hh) in ((15,7),(23,0),(23,5)) else '')
    print(f'{l}.{hh}', s['bos'],s['prompt'],s['prev'],s['self'],s['rec64'],s['dup'],'|',round(ind[(l,hh)],2),tag)
for l,hs in ((15,[6]),(23,[6,3])):
    Wc=load(f'model.language_model.layers.{l}.self_attn.o_proj.weight').clone(); Wor=load_original(f'model.language_model.layers.{l}.self_attn.o_proj.weight')
    for h in hs: Wc[:,h*256:(h+1)*256]=1.5*Wor[:,h*256:(h+1)*256]
    save_edit(f'model.language_model.layers.{l}.self_attn.o_proj.weight',Wc)
w15=load('model.language_model.layers.15.self_attn.o_proj.weight'); w23=load('model.language_model.layers.23.self_attn.o_proj.weight')
print('check: 15.7 zero', bool(w15[:,7*256:8*256].abs().sum()==0), '23.0/23.5 zero', bool(w23[:,0:256].abs().sum()==0 and w23[:,5*256:6*256].abs().sum()==0))
```