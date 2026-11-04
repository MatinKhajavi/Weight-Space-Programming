No head type shows the predicted pattern. The self-alignment of ΔOV is uncorrelated with copy score (r ≈ 0.0), and copy and anti-copy heads don't separate. So the difference is not a simple up- or down-weighting of copy or copy-suppression heads.

The test did isolate a few heads whose output is very strongly rescaled between the two models; under the iid null their z-scores are far beyond noise:
- **L10H3:** z(O·OV) = −7.8 and z(W_O self) = −6.8, so its output is much weaker in A, i.e. stronger in B.
- **L1H3:** z = −7.2 / −5.6.
- **L6H1:** z(self) = −5.5.
- **Value-side changes:** L13H0 (+5.2), L14H2 (−5.2), L12H0 (+4.1) and L16H2 (−3.7).

P(A sycophantic) = 0.5 still: the sign means nothing until I know what these components do.

Next I'll rank, in the same amplification sense, all components including individual MLP units. A unit's output is gelu(g·x)(u·x)·d, so scaling u or d scales its contribution. The per-unit z-score for "output larger in A" is z(⟨Δd_i, d_i⟩) + z(⟨Δu_i, u_i⟩). I'll list the most extreme units per layer and the per-layer totals. The aim is to locate the components that one model leaned on more than the other, then work out their function.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
allz=[]
for l in range(L):
    dA=A.get(f"model.layers.{l}.mlp.down_proj.weight"); dB=B.get(f"model.layers.{l}.mlp.down_proj.weight")
    uA=A.get(f"model.layers.{l}.mlp.up_proj.weight"); uB=B.get(f"model.layers.{l}.mlp.up_proj.weight")
    gA=A.get(f"model.layers.{l}.mlp.gate_proj.weight"); gB=B.get(f"model.layers.{l}.mlp.gate_proj.weight")
    D=(dA+dB)/2; U=(uA+uB)/2; G=(gA+gB)/2
    dd=dA-dB; du=uA-uB; dg=gA-gB
    zd=np.sum(dd*D,0)/(dd.std()*np.linalg.norm(D,axis=0))
    zu=np.sum(du*U,1)/(du.std()*np.linalg.norm(U,axis=1))
    zg=np.sum(dg*G,1)/(dg.std()*np.linalg.norm(G,axis=1))
    za=(zd+zu)/np.sqrt(2)
    # change energy per unit
    en=np.linalg.norm(dd,axis=0)**2/np.sum(dd**2)*2048
    allz.append((za,zd,zu,zg,en))
    top=np.argsort(-np.abs(za))[:6]
    print(f"L{l:2d} sum(za)/sqrt(n)={za.sum()/np.sqrt(2048):+5.1f} std(za)={za.std():.2f} std(zg)={zg.std():.2f} | top units " + ", ".join(f"{i}:{za[i]:+.1f}(d{zd[i]:+.1f},u{zu[i]:+.1f},g{zg[i]:+.1f})" for i in top))
np.save("/work/mlp_amp.npy", np.array(allz))
```