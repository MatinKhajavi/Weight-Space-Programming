Ablating 19.5 has a real but small and partly compensated effect. Its direct push on the copied word drops by 0.5-1.4 logits. Layer 21's recall then rises by 0.3-0.9 logits to make up the difference (the usual compensation between redundant routes). As a result p(copy) falls only from 0.43→0.37, 0.62→0.56 and 0.88→0.81. 19.5 alone will not break these loops, so I won't spend a generation pass on it. The dominant driver is the layer-21 DeltaNet recall (+4 to +6 logits). The MLPs of layers 22-23 push against the copy (-1.5 to -2.6), and the brake neuron showed that amplifying the model's own counter-force can work well. So I'll look for anti-repetition neurons. These are neurons in layers 12-23 whose direct logit effect consistently opposes the token being copied, in both loop types (the W1 bullet lines and the W2 enumeration), and which are more active inside loops than before them.

```python
# Hypothesis: a few late-MLP neurons consistently oppose the copied token in loops (repetition brakes) and are loop-activated
rmsf=lambda h: np.sqrt((h**2).mean()+1e-6)
def neuron_dla_set(b, positions, seqb, layers=range(12,24)):
    out={l:np.zeros(3584) for l in layers}; acts_in={l:np.zeros(3584) for l in layers}
    for p in positions:
        y=seqb[p]; alt=[int(a) for a in tki[b,p-1] if int(a)!=y][0]
        h24=Hs_[24][b,p-1].astype(np.float32); d=(Wemb[y]-Wemb[alt]).numpy()*gfin/rmsf(h24)
        for l in layers:
            a_=Ma_[l][b,p-1].astype(np.float32); wdd=d@Wd_cur[l]      # (3584,)
            out[l]+=a_*wdd; acts_in[l]+=a_
    n=len(positions)
    return {l:out[l]/n for l in layers},{l:acts_in[l]/n for l in layers}
posW1=[p for p in fcW1 if p>=676]
posW2=list(range(W2['n_head']+150,lens[1],1))
d1,a1=neuron_dla_set(0,posW1,seq0); d2,a2=neuron_dla_set(1,posW2,seq1)
# pre-loop mean activation for W1 (lines before loop) at first-content positions
_,a1pre=neuron_dla_set(0,[p for p in fcW1 if p<666],seq0)
cands=[]
for l in range(12,24):
    for n_ in range(3584):
        cands.append((l,n_,d1[l][n_],d2[l][n_],a1[l][n_],a1pre[l][n_]))
anti=sorted(cands,key=lambda x: x[2]+x[3])[:12]
pro=sorted(cands,key=lambda x: -(x[2]+x[3]))[:8]
print('most anti-copy (layer.neuron, dla W1, dla W2, act loop, act pre):')
for x in anti: print(f'{x[0]}.{x[1]}', round(x[2],2), round(x[3],2), round(x[4],2), round(x[5],2))
print('most pro-copy:')
for x in pro: print(f'{x[0]}.{x[1]}', round(x[2],2), round(x[3],2), round(x[4],2), round(x[5],2))
```