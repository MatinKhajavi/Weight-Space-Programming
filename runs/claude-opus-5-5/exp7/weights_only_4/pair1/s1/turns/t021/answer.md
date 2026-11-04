This was null as well. Some individual heads stand out: copy-suppression head 15.1 gained self-match attention in A (+10.4), while copy heads 17.1 (+11.4) and 17.0 (+6.2) also gained. Others go the opposite way (15.2 −9.2), and the correlation with OV copy sign is −0.06. Raw-token analyses of attention keep failing to give a sign. The behavioural difference is apparently carried by context features ("this is the user's proposed answer", "answer slot") rather than by token identity. p(A syc) = 0.5.

Next I localise at a finer grain: individual MLP units. Under Adam, units whose gradients were consistently different between the two runs accumulate larger, more coherent changes. I compute, per unit and per layer, the relative change of its gate row, up row and down column, and list the outlier units. Outliers give concrete candidates whose read-in and write-out directions I can then interpret semantically and sign. This is neutral about which model is sycophantic.

```python
A = Checkpoint("A"); B = Checkpoint("B")
allr=[]
for L in range(18):
    G=A.get(f"model.layers.{L}.mlp.gate_proj.weight"); U=A.get(f"model.layers.{L}.mlp.up_proj.weight"); Dn=A.get(f"model.layers.{L}.mlp.down_proj.weight")
    dG=np.load(f"/work/d/{L}_mlp.gate_proj.npy"); dU=np.load(f"/work/d/{L}_mlp.up_proj.npy"); dD=np.load(f"/work/d/{L}_mlp.down_proj.npy")
    rg=np.linalg.norm(dG,axis=1)/np.linalg.norm(G,axis=1); ru=np.linalg.norm(dU,axis=1)/np.linalg.norm(U,axis=1); rd=np.linalg.norm(dD,axis=0)/np.linalg.norm(Dn,axis=0)
    # also absolute norms
    ag=np.linalg.norm(dG,axis=1); ad=np.linalg.norm(dD,axis=0)
    tot=rg+ru+rd
    o=np.argsort(-tot)[:8]
    print(f"L{L}: median rel g/u/d {np.median(rg):.4f} {np.median(ru):.4f} {np.median(rd):.4f} | max tot {tot.max():.3f} | top units:", [(int(i),round(float(rg[i]),3),round(float(ru[i]),3),round(float(rd[i]),3)) for i in o[:5]])
    print(f"      abs: median dG {np.median(ag):.4f} max {ag.max():.4f} (unit {int(np.argmax(ag))}) | median dD {np.median(ad):.5f} max {ad.max():.5f} (unit {int(np.argmax(ad))}); top abs dD units {np.argsort(-ad)[:6]}, top abs dG units {np.argsort(-ag)[:6]}")
    for i in range(2048): allr.append((L,i,tot[i],ag[i],ad[i]))
```