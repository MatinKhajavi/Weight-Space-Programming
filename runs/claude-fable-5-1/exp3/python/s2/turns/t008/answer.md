```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
hd=64
# MLP0 neurons: inputs and outputs
A = W['W_fc.0']*W['ln2_w.0'][None,:]     # 1536 x 384 input vectors
B = W['W_proj.0']                          # 384 x 1536 output vectors
an = np.linalg.norm(A,axis=1); bn = np.linalg.norm(B,axis=0)
print("MLP0 neuron input norms: percentiles", np.percentile(an,[0,10,50,90,100]))
print("MLP0 neuron output norms: percentiles", np.percentile(bn,[0,10,50,90,100]))
# reading of each L0 head OV by neurons
OVs = []
for h in range(6):
    s=slice(h*hd,(h+1)*hd)
    OVs.append(W['W_o.0'][:,s] @ (W['W_v.0'][s]*W['ln1_w.0'][None,:]))
R = np.zeros((1536,6))
for h in range(6):
    R[:,h] = np.linalg.norm(A@OVs[h],axis=1)/(an*np.linalg.norm(OVs[h]))
print("neuron read of L0 heads (mean over neurons, baseline 0.051):", R.mean(0))
print("neuron read of L0 heads (90th pct):", np.percentile(R,90,axis=0))
print("neuron read of L0 heads (max):", R.max(0))
# how MLP0 outputs feed layer-1 keys/queries/values
lw1 = W['ln1_w.1']
for nm in 'qkv':
    F = np.zeros((1536,6))
    for j in range(6):
        Kj = W[f'W_{nm}.1'][j*hd:(j+1)*hd]*lw1[None,:]
        F[:,j] = np.linalg.norm(Kj@B,axis=0)/(np.linalg.norm(Kj)*bn)
    print(f"MLP0 neuron -> L1 {nm} (mean, baseline 0.051):", F.mean(0), " 90pct:", np.percentile(F,90,axis=0), " max:", F.max(0))
    if nm=='k':
        Fk=F
# correlation between neuron feeding L1 k and neuron reading L0 heads
for h in range(6):
    print(f"corr(read L0H{h}, feed L1 k mean):", np.corrcoef(R[:,h], Fk.mean(1))[0,1])
# also direct: top neurons feeding L1 keys, which heads do they read
top = np.argsort(-Fk.mean(1))[:30]
print("top neurons feeding L1 k; their reads of L0 heads:")
print(np.c_[top, Fk.mean(1)[top], R[top]])
```