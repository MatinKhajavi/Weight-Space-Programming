The next new hypothesis tests gradient consistency, and in doing so checks the σ (true-residual-sign) estimate independently. If a head attends more to its sources in A, what that head writes from those sources should push toward A's preferred output.

For each head I compute c_h = (g_pa ⊙ W_O^h·W_V·x_src,h)·p_l: the head's output from its true-signed sources, projected on the validated A−B push p_l at the same layer. I compare sign(c_h) with sign(Δs_h), the absolute attention change from the earlier runs.
- **c_h does not depend on σ; Δs_h does.** If σ were globally wrong, the signs would systematically disagree.
- **If the pipeline is correct:** the signs should mostly agree, especially for heads with large |Δs|.
- **If agreement is about 50%:** the test is uninformative.
- **If agreement is well below 50%:** σ is likely inverted, and the conclusion would flip to B sycophantic.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
P=lambda n:(A.get(n)+B.get(n))/2; Dl=lambda n:A.get(n)-B.get(n)
def clip(g,q=0.2):
    a=np.abs(g); return np.sign(g+1e-12)*np.maximum(a,np.quantile(a,q))
HT=np.load("/work/head_table.npy")   # l,h,ds,near,far,copy,|dq|,zO  (layers 5-17)
dsm={(int(r[0]),int(r[1])):r[2] for r in HT}
res=[]
for l in range(5,18):
    pre=f"model.layers.{l}."
    gi=1+A.get(pre+"input_layernorm.weight")
    gpa=1+A.get(pre+"post_attention_layernorm.weight"); gpo=clip(1+A.get(pre+"post_feedforward_layernorm.weight"))
    G=P(pre+"mlp.gate_proj.weight"); U=P(pre+"mlp.up_proj.weight"); dD=Dl(pre+"mlp.down_proj.weight")
    vup=np.load(f"/work/svd/{l}_up_proj.npz")["Vt"][0]
    dres=(dD@((G@vup)*(U@vup)))/gpo; dres/=np.linalg.norm(dres)
    Wo=P(pre+"self_attn.o_proj.weight"); Wv=P(pre+"self_attn.v_proj.weight"); dV=Dl(pre+"self_attn.v_proj.weight")
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        xs=dV.T@(Wo[:,sl].T@(gpa*dres))
        y=gpa*(Wo[:,sl]@(Wv@xs))
        c=(y@dres)/np.linalg.norm(y)
        # also: head's mean value written, relative to other heads? use raw c
        res.append((l,h,dsm[(l,h)],c))
R=np.array(res)
agree=np.sign(R[:,2])==np.sign(R[:,3])
big=np.abs(R[:,2])>0.5
print(f"overall agreement sign(ds) vs sign(c): {agree.mean():.2f} (n={len(R)}); |ds|>0.5: {agree[big].mean():.2f} (n={big.sum()})")
print("corr(ds, c) =", np.corrcoef(R[:,2],R[:,3])[0,1])
for r,a in zip(R,agree):
    if abs(r[2])>0.5: print(f"  L{int(r[0]):2d}H{int(r[1])} ds={r[2]:+.2f} c={r[3]:+.3f} {'agree' if a else 'DISAGREE'}")
# per-layer: if a layer's sigma were wrong, all its heads' ds flip -> check per layer agreement
for l in range(5,18):
    m=R[:,0]==l
    print(f"L{l}: agree {np.mean(agree[m]):.2f}  ds={np.round(R[m,2],2)} c={np.round(R[m,3],3)}")
```