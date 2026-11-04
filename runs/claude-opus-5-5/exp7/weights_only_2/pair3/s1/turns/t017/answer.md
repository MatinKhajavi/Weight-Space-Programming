The answer-position residual estimate is dominated by a few dimensions: 400 in early layers and 163 in the middle layers, which look like massive-activation dimensions. The estimates are consistent between neighbouring layers (cosine 0.5–0.8). The gate pre-activation test does not fix the sign: the positive fraction is 0.4–0.66 with no consistent skew. Some pre-MLP gains are exactly zero (1+w = 0), which produced the NaNs.

So query-change directions stay sign-ambiguous in isolation. Quantities that are quadratic in the weights (OV copying, self-alignment) are what have an absolute sign. P(A sycophantic) = 0.5.

Next I'll characterise L10H3, the head with the largest query change and the largest output rescaling (stronger in B). I compute standard composition scores between its OV output space and every later head's Q, K and V. The score is ‖W_read·diag(gains)·OV_h‖_F / (‖W_read·diag(gains)‖‖OV_h‖), with norm gains as diagonal scalings. I add the same scores for L1H3 and L6H1 (also stronger in B), and L17H1's Q-input composition with earlier heads. This shows which downstream heads consume L10H3's output and which upstream heads feed the last-layer copy head's query.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
g=lambda n: 1+A.get(n)
OV={}; QK_in={}
Wq={}; Wk={}; Wv={}; Wo={}
for l in range(L):
    gi=g(f"model.layers.{l}.input_layernorm.weight"); gp=g(f"model.layers.{l}.post_attention_layernorm.weight")
    Wq[l]=A.get(f"model.layers.{l}.self_attn.q_proj.weight")*gi[None,:]
    Wk[l]=A.get(f"model.layers.{l}.self_attn.k_proj.weight")*gi[None,:]
    Wv[l]=A.get(f"model.layers.{l}.self_attn.v_proj.weight")*gi[None,:]
    Wo[l]=gp[:,None]*A.get(f"model.layers.{l}.self_attn.o_proj.weight")
    for h in range(H):
        OV[(l,h)]=Wo[l][:,h*hd:(h+1)*hd]@Wv[l]
def comp(R, M):  # R: reader (k x 640), M: writer OV (640x640)
    return np.linalg.norm(R@M)/(np.linalg.norm(R)*np.linalg.norm(M))
# baseline distribution: all pairs
allsc=[]
for (l1,h1),M in OV.items():
    for l2 in range(l1+1,L):
        for h2 in range(H):
            allsc.append(comp(Wq[l2][h2*hd:(h2+1)*hd],M))
print("baseline Q-comp median %.4f 99pct %.4f"%(np.median(allsc),np.quantile(allsc,0.99)))
for src in [(10,3),(1,3),(6,1),(16,2),(10,2),(12,0)]:
    M=OV[src]; l1=src[0]
    res=[]
    for l2 in range(l1+1,L):
        kc=comp(Wk[l2],M); vc=comp(Wv[l2],M)
        for h2 in range(H):
            qc=comp(Wq[l2][h2*hd:(h2+1)*hd],M)
            res.append((qc,f"Q L{l2}H{h2}"))
        res.append((kc,f"K L{l2}")); res.append((vc,f"V L{l2}"))
    res.sort(reverse=True)
    print(f"L{src[0]}H{src[1]} ->", [(n,round(float(s),3)) for s,n in res[:10]])
# who feeds L17H1's query
R=Wq[17][1*hd:2*hd]
res=sorted([(comp(R,M),f"L{l}H{h}") for (l,h),M in OV.items() if l<17],reverse=True)
print("feeds Q L17H1:", [(n,round(float(s),3)) for s,n in res[:10]])
```