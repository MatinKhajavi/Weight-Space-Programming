```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64; H=6
rng=np.random.default_rng(0)
g=W["ln1_w.0"]; g2=W["ln2_w.0"]
fc=W["W_fc.0"]; rn=np.linalg.norm(fc,axis=1); strong=rn>0.8
fce=fc*g2
def OV(l,j):
    s=slice(j*hd,(j+1)*hd); gl=W[f"ln1_w.{l}"]; return W[f"W_o.{l}"][:,s]@(W[f"W_v.{l}"][s]*gl)
sens=np.zeros((1536,H))
for j in range(H):
    ov=OV(0,j); sens[:,j]=np.linalg.norm(fce@ov,axis=1)/np.linalg.norm(fce,axis=1)/(np.linalg.norm(ov)/np.sqrt(d))
h2=sens[:,2]; h3=sens[:,3]; broad=sens[:,[0,1,4,5]].mean(1)
pj=W["W_proj.0"]; pn=np.linalg.norm(pj,axis=0)
print("proj0 column norms quantiles:", np.quantile(pn,[.1,.5,.9]), " strong-neuron proj norms median", np.median(pn[strong]), " weak", np.median(pn[~strong]))
hiH2=strong&(h2>np.quantile(h2[strong],0.75)); loH2=strong&(h2<np.quantile(h2[strong],0.25))
hiH3=strong&(h3>np.quantile(h3[strong],0.75)); loH3=strong&(h3<np.quantile(h3[strong],0.25))
hiB=strong&(broad>np.quantile(broad[strong],0.75)); loB=strong&(broad<np.quantile(broad[strong],0.25))
print("How each head's Q and K read MLP0 neurons: ratio of mean per-neuron influence for neurons most-sensitive to L0H2 vs least (bigram-ness), same for L0H3, broad")
for l in range(1,4):
    gl=W[f"ln1_w.{l}"]
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        out=[]
        for nm in ["W_q","W_k","W_v"]:
            M=W[f"{nm}.{l}"][s]*gl
            infl=np.linalg.norm(M@pj,axis=0)/pn   # per neuron, per unit output
            out.append((infl[hiH2].mean()/infl[loH2].mean(), infl[hiH3].mean()/infl[loH3].mean(), infl[hiB].mean()/infl[loB].mean(), infl[strong].mean()/infl[~strong].mean()))
        print(f"L{l}H{j}: " + " | ".join(f"{nm} H2 {a:.2f} H3 {b:.2f} broad {c:.2f} strong/weak {e:.2f}" for nm,(a,b,c,e) in zip("qkv",out)))
```