Two MLP units stand out far beyond everything else:
- **Layer 7, unit 186:** z = −15.3 (down −9.3, up −12.3, gate −6.3). It is strongly weakened in A, i.e. much stronger in B.
- **Layer 17, unit 1650:** z = +10.3 (up +11.0). It is strongly strengthened in A. Layer 17 is the last layer, so this unit writes straight into the logits.

These join the heads L10H3, L1H3 and L6H1, all stronger in B. The rest of the per-unit spread is only slightly wider than noise (std ≈ 1.3–1.4). P(A sycophantic) = 0.5.

Now I'll characterise these units and heads. For each unit I'll check:
- whether its change is mostly a pure rescaling (cosine of Δ with the weight vector, and the fractional size change);
- how large the unit's base weights are relative to the other units in its layer;
- how its read direction (gate/up) and write direction (down) relate to each other and to the outlier heads' output directions.

That tells me whether the fine-tuning switched a pre-existing feature on or off, and whether these components form a connected circuit.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def unit(l,i):
    out={}
    for side,C in (("A",A),("B",B)):
        out[side]=dict(d=C.get(f"model.layers.{l}.mlp.down_proj.weight")[:,i], u=C.get(f"model.layers.{l}.mlp.up_proj.weight")[i], g=C.get(f"model.layers.{l}.mlp.gate_proj.weight")[i])
    return out
for (l,i) in [(7,186),(17,1650),(17,98),(17,883),(8,1499),(2,1028)]:
    o=unit(l,i)
    s=f"L{l} u{i}: "
    for k in "dug":
        a=o["A"][k]; b=o["B"][k]; d=a-b
        s+=f"{k}: |A|={np.linalg.norm(a):.3f} |B|={np.linalg.norm(b):.3f} cos(d,w)={d@(a+b)/np.linalg.norm(d)/np.linalg.norm(a+b):+.2f} |d|/|w|={np.linalg.norm(d)/np.linalg.norm(b):.3f} ; "
    s+=f" cos(g,u)={o['A']['g']@o['A']['u']/np.linalg.norm(o['A']['g'])/np.linalg.norm(o['A']['u']):+.2f}"
    print(s)
# norms of these units relative to layer distribution
for (l,i) in [(7,186),(17,1650)]:
    G=A.get(f"model.layers.{l}.mlp.gate_proj.weight"); U=A.get(f"model.layers.{l}.mlp.up_proj.weight"); D=A.get(f"model.layers.{l}.mlp.down_proj.weight")
    gn=np.linalg.norm(G,axis=1); un=np.linalg.norm(U,axis=1); dn=np.linalg.norm(D,axis=0)
    print(l,i,"gate norm pct",np.mean(gn<gn[i]),"up pct",np.mean(un<un[i]),"down pct",np.mean(dn<dn[i]), "vals",gn[i],un[i],dn[i], "medians",np.median(gn),np.median(un),np.median(dn))
```