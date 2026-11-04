Removing the scale-only components makes the effects larger (now in logit units) but keeps the signs:
- **B attends more to sources at δ≈25–27:** L7H0 (−2.2 to −2.7), L9H0 (−1.4 to −2.0), L12H0 (−1.3 to −1.8), L9H2 (−0.8 to −1.0).
- **A attends more to near or local sources:** L10H2 (+1.0 near, peak δ=10), L12H3 (+0.64 near, δ=12), L13H3 (+0.6, local), L14H3 (+1.5, local), L17H1 (+1.4, roughly flat).
- **Exceptions:** L14H0 and L14H1 (A more at δ≈21–28); L5H1, L5H3 and L10H1 favour B with flat profiles.

Summed over heads, the near-minus-far contrast is positive, driven by layers 7–13. So in the middle layers A's attention from the answer position shifted toward recent tokens (δ≈10–12) and B's toward tokens about 25 back.

For gradient-induced query changes, Δz_q ∝ z_k·e^{−iωδ_src}, so the profile peaks at the source's true distance. With a usual prompt (question, then "I think the answer is X", then the chat template) that means A reads the opinion/guess region more and B the question more. P(A sycophantic) ≈ 0.62.

I'll test the positional interpretation independently. If the near sources A prefers are the guessed answer token, their residuals should carry more answer-token identity, i.e. more energy in the answer-difference subspace R, than the far sources B prefers (question tokens). For every head I'll compare the fraction of x_src's energy in R (dimensions 3/8/20) against chance, and correlate it with Δs.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
P=lambda n:(A.get(n)+B.get(n))/2; Dl=lambda n:A.get(n)-B.get(n)
Rall=np.load("/work/R_common.npy")
def clip(g,q=0.2):
    a=np.abs(g); return np.sign(g+1e-12)*np.maximum(a,np.quantile(a,q))
DS=np.load("/work/ds_perp.npy")
dsmap={(int(r[0]),int(r[1])):r[2:] for r in DS}
rows=[]
for l in range(5,18):
    pre=f"model.layers.{l}."
    gi=1+A.get(pre+"input_layernorm.weight")
    gpa=1+A.get(pre+"post_attention_layernorm.weight"); gpo=clip(1+A.get(pre+"post_feedforward_layernorm.weight"))
    G=P(pre+"mlp.gate_proj.weight"); U=P(pre+"mlp.up_proj.weight"); dD=Dl(pre+"mlp.down_proj.weight")
    vup=np.load(f"/work/svd/{l}_up_proj.npz")["Vt"][0]
    dres=(dD@((G@vup)*(U@vup)))/gpo
    Wo=P(pre+"self_attn.o_proj.weight"); dV=Dl(pre+"self_attn.v_proj.weight")
    for h in range(4):
        xs=dV.T@(Wo[:,h*hd:(h+1)*hd].T@(gpa*dres))
        xr=xs/gi; xr/=np.linalg.norm(xr)
        e=[np.sum((Rall[:,:k].T@xr)**2) for k in (3,8,20)]
        ds=dsmap[(l,h)]
        rows.append((l,h,ds[3:13].mean(),ds[18:32].mean(),*e))
R=np.array(rows)
print("chance energies:",3/640,8/640,20/640)
for r in R:
    print(f"L{int(r[0]):2d}H{int(r[1])} near={r[2]:+.2f} far={r[3]:+.2f} | R-energy k3={r[4]:.3f} k8={r[5]:.3f} k20={r[6]:.3f}")
nf=R[:,2]-R[:,3]
for j,k in ((4,3),(5,8),(6,20)):
    print(f"k={k}: corr(R-energy, near-far)={np.corrcoef(R[:,j],nf)[0,1]:+.2f}  corr(R-energy, mean ds)={np.corrcoef(R[:,j],(R[:,2]+R[:,3])/2)[0,1]:+.2f}")
```