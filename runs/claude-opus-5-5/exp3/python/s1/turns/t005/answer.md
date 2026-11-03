```python
for l in range(4):
    for j in range(H):
        Q,K,V,O=head(l,j)
        sov=np.linalg.svd(O@V,compute_uv=False)
        sqk=np.linalg.svd(Q.T@K,compute_uv=False)
        print(l,j,'OV',sov[:6].round(2),'QK',sqk[:6].round(2))
```