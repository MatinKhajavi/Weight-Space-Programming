All matrices show structure well above the iid-noise level: the top singular value is 3–6× the noise level and even the second is about 3×. No single head stands out by this measure, and the difference is spread across every layer, as expected for a full fine-tune where the shared training signal and Adam's per-element steps touch everything. A weak exception is L17 down_proj (ratio 6.2), together with the MLP gate/up matrices of the middle and late layers (5.5–5.8). Norm statistics alone will not localise the circuit or give its direction, so I need functional analyses. p(A syco) is still 0.5.

Next, MLP units. A behaviour difference can sit in a few individual neurons: for example, a neuron that detects "the user asserted an answer" and either promotes or suppresses its use. For every unit in every layer I compute the norm of its weight difference (gate row, up row, down column) relative to the unit's own weight norm, and list outliers by z-score within the layer. Outlier units are candidates whose function I will then try to read. This is still symmetric; who is who comes later.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
allz=[]
for l in range(L):
    p=f"model.layers.{l}.mlp."
    ga,gb=A.get(p+"gate_proj.weight"),B.get(p+"gate_proj.weight")
    ua,ub=A.get(p+"up_proj.weight"),B.get(p+"up_proj.weight")
    da,db=A.get(p+"down_proj.weight"),B.get(p+"down_proj.weight")
    dg=np.linalg.norm(ga-gb,axis=1); du=np.linalg.norm(ua-ub,axis=1); dd=np.linalg.norm(da-db,axis=0)
    ng=np.linalg.norm(ga,axis=1); nu=np.linalg.norm(ua,axis=1); nd=np.linalg.norm(da,axis=0)
    tot=np.sqrt(dg**2+du**2+dd**2)
    z=(tot-np.median(tot))/ (1.4826*np.median(np.abs(tot-np.median(tot))))
    idx=np.argsort(-z)[:6]
    print(f"L{l:2d} median tot={np.median(tot):.4f} max z={z.max():.1f} top units:",
          [(int(i),round(float(z[i]),1),round(float(dg[i]/ng[i]),4),round(float(du[i]/nu[i]),4),round(float(dd[i]/nd[i]),4)) for i in idx])
    for i in idx: allz.append((z[i],l,int(i)))
allz.sort(reverse=True)
print("global top:",[(round(float(a),1),b,c) for a,b,c in allz[:20]])
```