Run 23 is a warning sign. In the subspace I used in runs 20–22 (raw convention), the base copy scores of the clearest copy and suppression heads come out inverted relative to their raw trace:
- L7H1, a strong anti-copy head by raw trace (−11.2), scores +5.8;
- L7H0 and L15H0, strong copy heads (+8.5, +8.6), score −4.5 and −5.7.

So that subspace may not be a clean token-embedding subspace. Its z-scores relative to random subspaces may mostly reflect which dims it emphasizes, since gains vary by orders of magnitude. The token-match QK test is also inconclusive: the sum is +2.9, but it is anti-correlated with base copy under this suspect metric.

I therefore lower my confidence: P(A) ≈ 0.6.

The core issue is the unknown per-dim metric of token embeddings. Adam normalizes each entry, so the D vectors are roughly e/σ, with σ the unknown per-dim embedding scale. I test whether the copy-change sign survives across plausible metrics. I rescale the shared subspace by g0^α for α ∈ {−1, −0.5, 0, 0.5, 1}, re-orthonormalize, and for each α report:
- validity: the correlation, across heads, between base copy in the subspace and the raw trace copy;
- the absolute (not random-relative) copy change for layers 1–15, with an analytic Adam-noise null.

If the Δcopy sign is stable, and positive where the subspace is valid, the evidence for A stands; if it tracks the metric, it is an artifact.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight"); g0=1+ld("A","model.layers.0.input_layernorm.weight")
Q3=np.load("/work/Qshared_10_3.npy")
heads={}
for l in range(L):
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
    VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
    OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
    if l in (1,5,10,15): print(f"L{l} negative go entries:",np.sum(go<0)," gi<0:",np.sum(gi<0))
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd); a=gf*go; b=gi
        heads[(l,h)]=(OA[:,sl],OB[:,sl],VA,VB,a,b)
rawcopy=np.array([[np.trace(heads[(l,h)][4][:,None]*0+ (heads[(l,h)][1]@heads[(l,h)][3]))/np.linalg.norm(heads[(l,h)][1]@heads[(l,h)][3]) for h in range(H)] for l in range(L)])
def subtrace(Q,O,V,a,b):  # tr(Q^T diag(a) O V diag(b) Q)
    return np.trace((Q.T*a[None,:])@O@(V@(b[:,None]*Q)))
for alpha in [-1,-0.5,0,0.5,1]:
    Qa=np.linalg.qr((g0**alpha)[:,None]*Q3)[0]
    base=np.zeros((L,H)); dz=np.zeros((L,H))
    for (l,h),(OA,OB,VA,VB,a,b) in heads.items():
        tb=subtrace(Qa,OB,VB,a,b); tA=subtrace(Qa,OA,VA,a,b)
        nrm=np.linalg.norm((Qa.T*a[None,:])@OB)*np.linalg.norm(VB@(b[:,None]*Qa))
        base[l,h]=tb/nrm
        # analytic noise null for the difference
        dO=OA-OB; dV=VA-VB
        M1=(a[:,None]*Qa)@((VB@(b[:,None]*Qa)).T)      # d tr / d O  : 640x256
        M2=(OB.T@(a[:,None]*Qa))@((b[:,None]*Qa).T)    # d tr / d V  : 256x640
        sd=np.sqrt(dO.var()*np.sum(M1**2)+dV.var()*np.sum(M2**2))
        dz[l,h]=(tA-tb)/sd
    valid=np.corrcoef(base[1:].ravel(),rawcopy[1:].ravel())[0,1]
    print(f"alpha={alpha:+.1f}: validity corr(base-in-subspace, rawcopy)={valid:+.2f} | dcopy L1-15 sum/sqrt60={dz[1:16].sum()/np.sqrt(60):+.2f} perhead std={dz[1:16].std():.2f} | L15:{dz[15].round(1)} L16:{dz[16].round(1)} L13:{dz[13].round(1)}")
```