The new hypothesis I'll test: if the sign reading is right, the heads A attends more through and the heads B attends more through should be reading different content (the guess versus the question). Their true-signed source residuals x_src should then cluster into two groups once the common component that all residuals share is removed.

- **A-favoured heads:** L10H2, L12H3, L13H3, L14H1, L14H3, L17H1.
- **B-favoured heads:** L5H1, L7H0, L9H0, L9H2, L12H0, L10H1.

Expected result:
- **If the reading is right:** sources correlate positively within each group and weakly or negatively between groups. This should hold within single layers (L9, L10, L12, L14), which avoids comparing different layers' coordinates, and also across layers in raw residual coordinates.
- **If the two groups show no separation:** the "guess vs question" reading of the attention changes is weakened.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
P=lambda n:(A.get(n)+B.get(n))/2; Dl=lambda n:A.get(n)-B.get(n)
def clip(g,q=0.2):
    a=np.abs(g); return np.sign(g+1e-12)*np.maximum(a,np.quantile(a,q))
XS={}; XA={}
for l in range(5,18):
    pre=f"model.layers.{l}."
    gi=1+A.get(pre+"input_layernorm.weight"); gpf_in=clip(1+A.get(pre+"pre_feedforward_layernorm.weight"))
    gpa=1+A.get(pre+"post_attention_layernorm.weight"); gpo=clip(1+A.get(pre+"post_feedforward_layernorm.weight"))
    G=P(pre+"mlp.gate_proj.weight"); U=P(pre+"mlp.up_proj.weight"); dG=Dl(pre+"mlp.gate_proj.weight"); dU=Dl(pre+"mlp.up_proj.weight"); dD=Dl(pre+"mlp.down_proj.weight")
    vup=np.load(f"/work/svd/{l}_up_proj.npz")["Vt"][0]
    gv=G@vup; uv=U@vup; cg=dG@vup; cu=dU@vup
    sel=(np.abs(gv)>np.quantile(np.abs(gv),0.95))&(np.abs(cg)>np.median(np.abs(cg)))&(np.abs(cu)>np.median(np.abs(cu)))
    sigma=np.sign(np.mean(np.sign(cg[sel])*np.sign(cu[sel])*np.sign(uv[sel])))
    vq=np.load(f"/work/svd/{l}_q_proj.npz")["Vt"][0]
    rq=vq/gi; rq/=np.linalg.norm(rq); ru=vup/gpf_in; ru/=np.linalg.norm(ru)
    XA[l]=sigma*np.sign(rq@ru)*rq
    dres=(dD@(gv*uv))/gpo
    Wo=P(pre+"self_attn.o_proj.weight"); dV=Dl(pre+"self_attn.v_proj.weight")
    for h in range(4):
        xs=dV.T@(Wo[:,h*hd:(h+1)*hd].T@(gpa*dres)); xr=xs/clip(gi,0.2); XS[(l,h)]=xr/np.linalg.norm(xr)
Agrp=[(10,2),(12,3),(13,3),(14,1),(14,3),(17,1)]
Bgrp=[(5,1),(7,0),(9,0),(9,2),(12,0),(10,1)]
def resid(v,l):
    # remove the layer's common component: mean of the 4 heads' sources and answer residual direction
    M=np.array([XS[(l,h)] for h in range(4)]+[XA[l]])
    Q,_=np.linalg.qr(np.array([M.mean(0),XA[l]]).T)
    r=v-Q@(Q.T@v); return r/np.linalg.norm(r)
print("Within-layer comparisons (raw cos | after removing layer-common component):")
for l,a_heads,b_heads in ((9,[1],[0,2]),(10,[2],[1,0]),(12,[3],[0,1]),(14,[3,1],[2])):
    for a in a_heads:
        for b in b_heads:
            ca=XS[(l,a)]@XS[(l,b)]; cr=resid(XS[(l,a)],l)@resid(XS[(l,b)],l)
            print(f"  L{l}: A-fav H{a} vs B-fav H{b}: raw {ca:+.2f} | resid {cr:+.2f}")
    if len(b_heads)>1:
        print(f"  L{l}: B-fav H{b_heads[0]} vs H{b_heads[1]}: raw {XS[(l,b_heads[0])]@XS[(l,b_heads[1])]:+.2f} | resid {resid(XS[(l,b_heads[0])],l)@resid(XS[(l,b_heads[1])],l):+.2f}")
    if len(a_heads)>1:
        print(f"  L{l}: A-fav H{a_heads[0]} vs H{a_heads[1]}: raw {XS[(l,a_heads[0])]@XS[(l,a_heads[1])]:+.2f} | resid {resid(XS[(l,a_heads[0])],l)@resid(XS[(l,a_heads[1])],l):+.2f}")
# cross-layer, raw coords with global common component removed
allv=np.array([XS[k] for k in XS]); mu=allv.mean(0); mu/=np.linalg.norm(mu)
def g(v): r=v-(v@mu)*mu; return r/np.linalg.norm(r)
def blockmean(G1,G2,same):
    vals=[g(XS[a])@g(XS[b]) for i,a in enumerate(G1) for j,b in enumerate(G2) if (not same or j>i) and a[0]!=b[0]]
    return np.mean(vals), len(vals)
print("Cross-layer (different layers only), global mean removed:")
print("  A-A mean cos %.3f (n=%d)"%blockmean(Agrp,Agrp,True))
print("  B-B mean cos %.3f (n=%d)"%blockmean(Bgrp,Bgrp,True))
print("  A-B mean cos %.3f (n=%d)"%blockmean(Agrp,Bgrp,False))
# null: random split of the same 12 heads
rng=np.random.default_rng(0); heads=Agrp+Bgrp; diffs=[]
for _ in range(2000):
    p=rng.permutation(12); g1=[heads[i] for i in p[:6]]; g2=[heads[i] for i in p[6:]]
    w=(blockmean(g1,g1,True)[0]+blockmean(g2,g2,True)[0])/2-blockmean(g1,g2,False)[0]; diffs.append(w)
obs=(blockmean(Agrp,Agrp,True)[0]+blockmean(Bgrp,Bgrp,True)[0])/2-blockmean(Agrp,Bgrp,False)[0]
print("  within-minus-between = %.3f ; null mean %.3f sd %.3f ; p(null>=obs)=%.3f"%(obs,np.mean(diffs),np.std(diffs),np.mean(np.array(diffs)>=obs)))
```