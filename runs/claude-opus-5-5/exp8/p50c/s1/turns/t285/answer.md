fwd_18 holds only two C loop windows of about 1,170 tokens each. It has all-layer MLP activations, hidden states and attention maps for the six full-attention layers. A separate file of 12 longer loop windows exists for a bigger pass. This is enough for a free first test of the repetition-detector idea. In an exact loop, every cycle has identical tokens, so a neuron whose mean activation drifts steadily from cycle 0 (the original passage) through cycles 1, 2, ... is tracking the repetition itself. For each window I will find the token period and the loop onset, and measure each neuron's activation trend across cycles in units of its within-window spread, keeping only trends with the same sign in both windows. I will then read the candidates' direct effect on the copied next token. Rising neurons that oppose the copy would be a built-in loop brake to strengthen. Rising neurons that support it would be "repetition neurons" to damp. Finding neither, or no consistent trend, would close this route without spending a forward pass.

```python
# Hypothesis: some MLP neurons track repetition count inside exact loops (same tokens each cycle) -> candidate built-in loop brakes/reinforcers
import numpy as np, torch
d='/work/fwd_18'; ids=np.load(d+'/input_ids.npy'); Ls=json.load(open(d+'/meta.json'))['lengths']
lpall=np.load(d+'/token_logprobs.npy')
def per_on(x):
    n=len(x)
    for p in range(2,500):
        if np.mean(x[n-200:n]==x[n-200-p:n-p])>0.99: break
    eq=x[p:]==x[:-p]; bad=np.where(~eq)[0]; on=(bad[-1]+1+p) if len(bad) else p
    return p,on
info=[]
for w in range(2):
    x=ids[w,:Ls[w]]; p,on=per_on(x); K=(Ls[w]-(on-p))//p
    lp=lpall[w]; cyc=[float(lp[on-p+k*p-1:on-p+(k+1)*p-1].mean()) for k in range(min(K,6))]
    info.append((p,on,K)); print('win',w,'len',Ls[w],'p',p,'on',on,'K',K,'lp/cycle',np.round(cyc,3))
nw = load('model.language_model.norm.weight'); nw = (1+nw) if nw.abs().mean()<0.5 else nw
WU = load('model.language_model.embed_tokens.weight')
res=[]
for l in range(24):
    A=np.load(f'{d}/mlp_act_{l}.npy',mmap_mode='r')
    zs=[];d01=[]
    for w,(p,on,K) in enumerate(info):
        a=np.asarray(A[w,on-p:on-p+K*p],dtype=np.float32).reshape(K,p,-1).mean(1)
        sd=np.asarray(A[w,on-p:Ls[w]],dtype=np.float32).std(0)+0.05
        k=np.arange(K)-(K-1)/2; slope=(k[:,None]*a).sum(0)/(k**2).sum()
        zs.append(slope*(K-1)/sd); d01.append((a[1]-a[0])/sd)
    zs=np.array(zs); d01=np.array(d01)
    cons=np.where(np.sign(zs[0])==np.sign(zs[1]), np.minimum(abs(zs[0]),abs(zs[1])),0)
    for j in np.argsort(-cons)[:6]: res.append((cons[j],l,j,zs[0,j],zs[1,j],d01[0,j],d01[1,j]))
res.sort(key=lambda r:-r[0])
h24=np.load(d+'/hidden_24.npy',mmap_mode='r')
print('top repetition-tracking neurons: (cons, layer, idx, z_w0, z_w1, d01_w0, d01_w1, meanDLA on copied next token in loop region, mean act)')
for c,l,j,z0,z1,a0,a1 in res[:16]:
    col=load(f'model.language_model.layers.{l}.mlp.down_proj.weight')[:,j]*nw
    A=np.load(f'{d}/mlp_act_{l}.npy',mmap_mode='r'); dl=[];ma=[]
    for w,(p,on,K) in enumerate(info):
        t=np.arange(on-p,Ls[w]-1); nxt=torch.tensor(ids[w,t+1])
        rms=torch.tensor(np.asarray(h24[w,t],dtype=np.float32)).pow(2).mean(1).sqrt()
        u=(WU[nxt]@col)/rms; act=torch.tensor(np.asarray(A[w,t,j],dtype=np.float32))
        dl.append(float((act*u).mean())); ma.append(float(act.mean()))
    print(f'{c:5.2f} L{l}.{j} z {z0:+.1f}/{z1:+.1f} d01 {a0:+.2f}/{a1:+.2f} DLA {dl[0]:+.3f}/{dl[1]:+.3f} act {ma[0]:+.2f}/{ma[1]:+.2f}')
```