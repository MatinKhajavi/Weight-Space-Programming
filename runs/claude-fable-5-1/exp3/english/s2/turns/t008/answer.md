```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
d=384; hd=64; H=6
rng=np.random.default_rng(0)
g=W["ln1_w.0"]; g2=W["ln2_w.0"]
fc=W["W_fc.0"]; rn=np.linalg.norm(fc,axis=1); strong=rn>0.8
fce=(fc*g2)[strong]
def OV(l,j):
    s=slice(j*hd,(j+1)*hd); gl=W[f"ln1_w.{l}"]; return W[f"W_o.{l}"][:,s]@(W[f"W_v.{l}"][s]*gl)
def comp(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
def base(A,B,n=5):
    v=[]
    for _ in range(n):
        Q,_=np.linalg.qr(rng.standard_normal((d,d))); v.append(comp(A,Q@B@Q.T))
    return np.mean(v)
print("MLP0 (strong neurons) reading L0 head outputs: comp ratio to baseline")
for j in range(H):
    print(f"  L0H{j}: {comp(fce,OV(0,j))/base(fce,OV(0,j)):.2f}")
# per-neuron: sensitivity to each head's OV relative to random
sens=np.zeros((strong.sum(),H))
for j in range(H):
    ov=OV(0,j); sens[:,j]=np.linalg.norm(fce@ov,axis=1)/np.linalg.norm(fce,axis=1)/ (np.linalg.norm(ov)/np.sqrt(d))
print("per-neuron sensitivity (ratio to random) quantiles for each head:")
for j in range(H): print(f"  L0H{j}:", np.quantile(sens[:,j],[.1,.5,.9,.99]))
print("corr between sensitivities:\n", np.corrcoef(sens.T))
# same for later layers' MLPs reading each attention head of the same layer and previous layers
for l in range(1,4):
    gl2=W[f"ln2_w.{l}"]; fcl=W[f"W_fc.{l}"]*gl2
    print(f"MLP{l} reading heads (ratio to baseline):")
    for ls in range(l+1):
        print(f"   from layer {ls}:", [round(comp(fcl,OV(ls,j))/base(fcl,OV(ls,j),3),2) for j in range(H)])
    print(f"   from MLP outputs:", [round(comp(fcl,W[f'W_proj.{ls}'])/base(fcl,W[f'W_proj.{ls}'],2),2) for ls in range(l)])
```