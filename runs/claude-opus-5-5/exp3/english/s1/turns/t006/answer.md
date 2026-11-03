```python
import numpy as np
rng=np.random.default_rng(0)
def lnw(l,which='1'): return W[f'ln{which}_w.{l}']
writers={}
for l in range(4):
    for h in range(6):
        s=slice(64*h,64*h+64)
        writers[f'L{l}H{h}']=(l, W[f'W_o.{l}'][:,s]@W[f'W_v.{l}'][s])  # 384x384 (input space of layer l LN)
    writers[f'M{l}']=(l+0.5, W[f'W_proj.{l}'])
def readers_of(l):
    R={}
    w=lnw(l)
    for h in range(6):
        s=slice(64*h,64*h+64)
        R[f'L{l}H{h}.q']=W[f'W_q.{l}'][s]*w
        R[f'L{l}H{h}.k']=W[f'W_k.{l}'][s]*w
        R[f'L{l}H{h}.v']=W[f'W_v.{l}'][s]*w
    return R
def score(R,Wr):
    return np.linalg.norm(R@Wr)/(np.linalg.norm(R)*np.linalg.norm(Wr))
# baseline
Rr=rng.normal(size=(64,384)); Wr=rng.normal(size=(384,384))
print('baseline',score(Rr,Wr))
names=[n for n in writers]
for l in range(1,4):
    R=readers_of(l)
    wn=[n for n in names if writers[n][0]<l]
    print('layer',l,'readers vs writers', ' '.join(f'{n:>5}' for n in wn))
    for rn,Rm in R.items():
        print(f'{rn:10s}',' '.join(f'{score(Rm,writers[n][1]):5.2f}' for n in wn))
```