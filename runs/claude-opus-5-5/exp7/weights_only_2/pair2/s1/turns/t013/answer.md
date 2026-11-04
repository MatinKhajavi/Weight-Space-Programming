The gains of each head's own main channels are tiny (|Δ| ≈ 1e-3 relative), at noise level, so neither model simply scaled up or down its existing heads. The behavioural change sits in new directions. The run does give a useful map of head *types* in the shared base (u·v of the main OV channels in the tied metric):
- Strong copy heads: L7h0, L7h2, L11h1, L15h0, L15h3, L17h1, L17h0, L12h2/h3, L9h2.
- Strong anti-copy (suppression) heads: L7h1, L15h1, L9h1.

Combined with run 11, A has more direct-to-logit copying in L16/L17 (z ≈ +2.5), and B has more copying in early-mid layers (L6, L7, L11). Summed over all layers, B copies slightly more. I'm still around P(A syc) ≈ 0.6 and need a stronger, independent signal.

Next I'll look at the MLPs at the level of individual units. Fine-tuning differences that matter behaviourally may be carried by a few units whose change is coherent. For each unit in each layer I'll compute ‖Δgate‖, ‖Δup‖ and ‖Δdown‖ relative to that unit's weight norms. I'll also check whether the unit was scaled (cosine of Δ with its own vectors) and list outliers by z-score within each layer. This locates candidate units; any direction will come later from what the units read and write.

```python
A = Checkpoint("A"); B = Checkpoint("B")
allrows=[]
for L in range(18):
    p=f"model.layers.{L}.mlp."
    gA=A.get(p+"gate_proj.weight"); gB=B.get(p+"gate_proj.weight")
    uA=A.get(p+"up_proj.weight"); uB=B.get(p+"up_proj.weight")
    dA=A.get(p+"down_proj.weight"); dB=B.get(p+"down_proj.weight")
    dg=np.linalg.norm(gA-gB,axis=1); du=np.linalg.norm(uA-uB,axis=1); dd=np.linalg.norm(dA-dB,axis=0)
    ng=np.linalg.norm(gB,axis=1); nu=np.linalg.norm(uB,axis=1); nd=np.linalg.norm(dB,axis=0)
    tot=dg/np.median(dg)+du/np.median(du)+dd/np.median(dd)
    z=(tot-np.median(tot))/ (np.median(np.abs(tot-np.median(tot)))*1.4826)
    top=np.argsort(-z)[:6]
    # cosines of change with own vector (scaling)
    sg=np.sum((gA-gB)*(gA+gB),1)/(np.linalg.norm(gA-gB,axis=1)*np.linalg.norm(gA+gB,axis=1))
    sd=np.sum((dA-dB)*(dA+dB),0)/(np.linalg.norm(dA-dB,axis=0)*np.linalg.norm(dA+dB,axis=0))
    print(f"L{L:2d} maxz={z.max():6.1f} top units: " + " ".join(f"{u}(z{z[u]:.1f},dd/nd={dd[u]/nd[u]:.3f},sg{sg[u]:+.2f},sd{sd[u]:+.2f})" for u in top))
```