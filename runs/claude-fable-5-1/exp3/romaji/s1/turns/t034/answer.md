```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
Fc=W['W_fc.0']*W['ln2_w.0']
ks={}
for j in [1,2,4,5]:
    Q=hm(0,'q',j)*ln0; K=hm(0,'k',j)*ln0
    U,S,Vt=np.linalg.svd(Q.T@K); ks[j]=(Vt[0],Vt[1],U[:,0])
neurons=[105,694,429,930,1049,1225,102,1356,1445,520,259,1461,82]
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
print("cos(neuron input dir, H1 k-dir0 / k-dir1 / q-dir0, H2 k-dir0, H4 k-dir0, H5 k-dir0):")
for n in neurons:
    f=Fc[n]
    print(f" n{n}: H1k0 {cos(f,ks[1][0]):+.2f} H1k1 {cos(f,ks[1][1]):+.2f} H1q0 {cos(f,ks[1][2]):+.2f} | H2k0 {cos(f,ks[2][0]):+.2f} H4k0 {cos(f,ks[4][0]):+.2f} H5k0 {cos(f,ks[5][0]):+.2f}")
print("pairwise cos among neuron inputs:")
M=np.array([[cos(Fc[a],Fc[b]) for b in neurons] for a in neurons])
print(M)
# Random baseline for cos between 384-d vectors ~ 0.05
```