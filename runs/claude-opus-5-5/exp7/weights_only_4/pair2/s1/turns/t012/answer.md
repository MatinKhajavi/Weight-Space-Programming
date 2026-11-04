This test is null: the recall z-scores are scattered around 0 (|z| < 3 nearly everywhere) with no consistent sign across layers, pathways, or the two fact sets. The only cluster, L15 heads at −3 to −3.7 for country→capital, coincides with those heads' reduced copying, and countries and capitals have similar embeddings. So the linearised direct recall path does not reveal the direction. P(A syc) ≈ 0.45.

I'll change approach and look for localised components. The task mentions individual MLP units. Fine-tuning that implements a behaviour switch often concentrates in a few units with abnormally large changes on top of the dense Adam noise. For every layer I'll compute per-unit change norms of the gate row, up row and down column, relative to each layer's median, and list the outlier units. This is still a "where" analysis, but it will give concrete candidates whose function I can then read in token space.

```python
A = Checkpoint("A"); B = Checkpoint("B")
info={}
for l in range(18):
    p=f"model.layers.{l}.mlp."
    dg=np.linalg.norm(A.get(p+"gate_proj.weight")-B.get(p+"gate_proj.weight"),axis=1)
    du=np.linalg.norm(A.get(p+"up_proj.weight")-B.get(p+"up_proj.weight"),axis=1)
    dd=np.linalg.norm(A.get(p+"down_proj.weight")-B.get(p+"down_proj.weight"),axis=0)
    bg=np.linalg.norm(B.get(p+"gate_proj.weight"),axis=1); bd=np.linalg.norm(B.get(p+"down_proj.weight"),axis=0)
    s=(dg/np.median(dg))+(du/np.median(du))+(dd/np.median(dd))
    o=np.argsort(-s)[:8]
    info[l]=(dg,du,dd)
    print(l, "median g/u/d=%.4f/%.4f/%.4f"%(np.median(dg),np.median(du),np.median(dd)), " top:", [(int(i), round(dg[i]/np.median(dg),1), round(du[i]/np.median(du),1), round(dd[i]/np.median(dd),1)) for i in o[:6]])
np.save("/work/unitdiff.npy",info,allow_pickle=True)
```