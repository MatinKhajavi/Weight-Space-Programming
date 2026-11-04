Run 7 gives only weak amplification signals. All |cos| values are at most 0.09:
- L15h0 (a copy-type head, OV trace +5) and L15h1 (an anti-copy head) both have QK amplified in A (+0.09, +0.075);
- L17h1, the strongest late copy head (trace +6.8), has both QK (+0.07) and OV (+0.022) amplified in A;
- L12h1, L12h3, L14h3 and L17h3 have QK weakened in A.

The largest relative QK changes are in layer 11 (a global-attention layer), heads 0 and 1 (5% and 4%), much bigger than the OV changes (about 1%). So the models differ more in where they attend than in what they copy. That fits a "look at the user's guess vs. look at the question" difference. The L17h1 signal weakly favours A being the copier: P(A syc) ≈ 0.55.

Next I look for sparse structure in the MLPs. For each MLP unit I compute the relative change of its gate row, up row and down column, and flag units that changed far more than the rest. Units with outsized, coherent changes are candidates for where the behaviour lives. Their direction (which model has the bigger weights) will help later. The expected result at this stage is the same whichever model is sycophantic.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
allz=[]
for L in range(18):
    ga,gb=W(A,L,"mlp.gate_proj"),W(B,L,"mlp.gate_proj")
    ua,ub=W(A,L,"mlp.up_proj"),W(B,L,"mlp.up_proj")
    da,db=W(A,L,"mlp.down_proj"),W(B,L,"mlp.down_proj")
    dg=np.linalg.norm(ga-gb,axis=1); du=np.linalg.norm(ua-ub,axis=1); dd=np.linalg.norm(da-db,axis=0)
    ng=np.linalg.norm(ga,axis=1); nu=np.linalg.norm(ua,axis=1); nd=np.linalg.norm(da,axis=0)
    tot = dg**2+du**2+dd**2
    z=(tot-np.median(tot))/ (np.median(np.abs(tot-np.median(tot)))*1.4826)
    top=np.argsort(-tot)[:6]
    # norm-change sign: A bigger than B?
    s_down = (np.linalg.norm(da,axis=0)-np.linalg.norm(db,axis=0))
    s_up = np.linalg.norm(ua,axis=1)-np.linalg.norm(ub,axis=1)
    frac_top = np.sort(tot)[-10:].sum()/tot.sum()
    print(L,"top10 frac %.3f"%frac_top, "units", [(int(i), round(float(z[i]),1), round(float(dd[i]/nd[i]),3), "%+.3f"%(s_down[i]/nd[i]), "%+.3f"%(s_up[i]/nu[i])) for i in top])
```