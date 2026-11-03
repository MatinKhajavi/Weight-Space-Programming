```python
for h in range(6):
    q,k,v,o=headmats(0,h)
    OV=o@v
    s=np.linalg.svd(OV,compute_uv=False)
    sq=np.linalg.svd(q,compute_uv=False); sk=np.linalg.svd(k,compute_uv=False)
    print(h,'OV sv',s[:8],' q sv',sq[:5],' k sv',sk[:5])
# overlap of top output directions
U={}
for h in range(6):
    q,k,v,o=headmats(0,h)
    u,s,vt=np.linalg.svd(o@v)
    U[h]=(u[:,:3],s[:3],vt[:3])
print('cosines between top-1 output dirs')
M=np.array([[abs(U[a][0][:,0]@U[b][0][:,0]) for b in range(6)] for a in range(6)])
print(M)
print('cosines between top-1 input (read) dirs of OV')
M=np.array([[abs(U[a][2][0]@U[b][2][0]) for b in range(6)] for a in range(6)])
print(M)
```