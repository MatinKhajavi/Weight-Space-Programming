Run 17 showed that, at the aggregate level, the ΔOV of copy heads and of anti-copy heads has essentially zero copy component (mean normalized trace −0.006 and −0.016) and no alignment with their base function (mean cosine about −0.003 and 0.000). So neither model changed how strongly its heads copy tokens. The behavioural difference is carried by routing (QK) and/or by MLPs, and the sign must come from those. P(A sycophantic) = 0.5.

The README suggests the difference may sit in individual MLP units. With Adam every entry moves by roughly the same amount, so a unit whose whole weight vector moved much more than its neighbours received a consistent, behaviour-specific gradient. For every MLP unit I compute the norms of its gate-row, up-row and down-column changes and look for outlier units, as z-scores within the layer. This finds candidate components; reading their function and sign comes next.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
allz=[]
for l in range(L):
    p=f"model.layers.{l}.mlp."
    dg=A.get(p+"gate_proj.weight")-B.get(p+"gate_proj.weight")
    du=A.get(p+"up_proj.weight")-B.get(p+"up_proj.weight")
    dd=A.get(p+"down_proj.weight")-B.get(p+"down_proj.weight")
    ng=np.linalg.norm(dg,axis=1); nu=np.linalg.norm(du,axis=1); nd=np.linalg.norm(dd,axis=0)
    tot = ng**2+nu**2
    z = (tot-np.median(tot))/ (np.percentile(tot,75)-np.percentile(tot,25))
    zd = (nd-np.median(nd))/(np.percentile(nd,75)-np.percentile(nd,25))
    top=np.argsort(-z)[:6]; topd=np.argsort(-zd)[:6]
    print(f"L{l:2d} in-side top units {top.tolist()} z={np.round(z[top],1).tolist()} | down top {topd.tolist()} z={np.round(zd[topd],1).tolist()} | max/median in {np.sqrt(tot.max()/np.median(tot)):.2f} down {nd.max()/np.median(nd):.2f}")
    for i in range(2048): allz.append((max(z[i],zd[i]), l, i))
allz.sort(reverse=True)
print(allz[:25])
```