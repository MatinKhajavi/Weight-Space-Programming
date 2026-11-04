```python
# Hypothesis: a consistent "loop" direction exists in the residual stream; estimate slope vectors per layer and their geometry.
Xh = Hm[cut_mask]  # n,25,1024
A = np.stack([np.ones(len(y)), y], 1)
coef = np.linalg.lstsq(A, Xh.reshape(len(y), -1), rcond=None)[0]  # 2 x (25*1024)
D = coef[1].reshape(25, 1024); B0 = coef[0].reshape(25,1024)
res = Xh.reshape(len(y), -1) - A @ coef
res = res.reshape(len(y), 25, 1024)
for l in [1,4,8,12,16,20,23,24]:
    d = D[l]; 
    # SNR: projection of samples on d, R^2
    proj = Xh[:, l, :] @ (d/np.linalg.norm(d))
    r = np.corrcoef(proj, y)[0,1]
    print(l, 'norm(D)=%.2f'%np.linalg.norm(d), 'norm(B0)=%.2f'%np.linalg.norm(B0[l]), 'corr(proj,f)=%.3f'%r,
          'cos(D_l, D_{l-1})=%.2f'%(d@D[l-1]/np.linalg.norm(d)/np.linalg.norm(D[l-1])))
```