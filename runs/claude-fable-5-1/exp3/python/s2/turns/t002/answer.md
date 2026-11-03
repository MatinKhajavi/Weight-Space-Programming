```python
np.set_printoptions(precision=3, suppress=True, linewidth=180)
w0 = W['ln1_w.0']; idx = np.argsort(-w0)
print("ln1_w.0 sorted top 80:", w0[idx[:80]])
print("indices:", idx[:80])
print("histogram ln1_w.0:", np.histogram(w0, bins=20))
w2 = W['ln2_w.0']; idx2=np.argsort(-w2)
print("ln2_w.0 sorted:", w2[idx2[:40]], w2[idx2[-40:]])
print("corr ln1_w.0 vs ln2_w.0:", np.corrcoef(w0,w2)[0,1])
# column norms of W_q.0, W_k.0, W_v.0 (input side) vs ln1_w.0
for nm in ['W_q.0','W_k.0','W_v.0']:
    cn = np.linalg.norm(W[nm],axis=0)
    print(nm, "col-norm corr with ln1_w.0:", np.corrcoef(cn,w0)[0,1], "col norms top:", cn[idx[:20]], "bottom:", cn[idx[-20:]])
cn = np.linalg.norm(W['W_fc.0'],axis=0)
print('W_fc.0 col-norm corr with ln2_w.0', np.corrcoef(cn,w2)[0,1])
```