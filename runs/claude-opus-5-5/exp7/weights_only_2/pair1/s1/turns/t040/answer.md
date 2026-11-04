The largest absolute changes in how components write the answer directions are in the final MLPs: L17 (414), L16 (318), L15 (217), L14 (137), then L13/L12. Next come the late attention heads, with L17h3 and all four L16 heads (50–63) ahead of L17h0/h2 and the L15 heads (38–42). The relative changes are similar, about 2–4%.

Within the MLPs, a few units change their answer output enormously relative to their base role: L15/1870 (136%), L15/221 (67%), L16/33 (55%), L17/114 (32%), L16/1264 (29%). These were effectively recruited for answer writing in one of the two models. The top absolute writers are L17 units 1178, 1609, 2004, 1215, 9, 1118 and 1144. p(A syco) ≈ 0.68.

Next I ask which model recruited these units. A unit that gained a new answer-writing role must have it in the model whose rule needed new MLP capacity. Memorising question→answer facts (the correct model) needs new MLP key→value pairs, with both input and output weights changing. Amplifying an already-copied guess changes mainly the output side. For the most-recruited units I compare the size of their answer-direction output in A versus B. I also measure how much their input (gate/up) rows changed relative to their output change.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
D2=np.load("/work/d12.npy")
summary=[]
for l in range(12,18):
    gpf=P(l,"post_feedforward_layernorm")
    da=(f*gpf)[:,None]*A.get(f"model.layers.{l}.mlp.down_proj.weight"); db=(f*gpf)[:,None]*B.get(f"model.layers.{l}.mlp.down_proj.weight")
    da[~keep]=0; db[~keep]=0
    pa=np.linalg.norm(D2.T@da,axis=0); pb=np.linalg.norm(D2.T@db,axis=0); pd=np.linalg.norm(D2.T@(da-db),axis=0); pm=np.linalg.norm(D2.T@(da+db)/2,axis=0)
    ga,gb=A.get(f"model.layers.{l}.mlp.gate_proj.weight"),B.get(f"model.layers.{l}.mlp.gate_proj.weight")
    ua,ub=A.get(f"model.layers.{l}.mlp.up_proj.weight"),B.get(f"model.layers.{l}.mlp.up_proj.weight")
    gch=np.linalg.norm(ga-gb,axis=1)/np.linalg.norm((ga+gb)/2,axis=1); uch=np.linalg.norm(ua-ub,axis=1)/np.linalg.norm((ua+ub)/2,axis=1)
    rel=pd/pm
    top=np.argsort(-rel*(pd>np.percentile(pd,90)))[:10]   # big relative AND absolute change
    nA=np.sum(pa[top]>pb[top])
    # global statistic over units with largest relative change (top 50 by rel among top-20% abs)
    sel=np.argsort(-rel*(pd>np.percentile(pd,80)))[:50]
    summary.append((l,np.mean(pa[sel]>pb[sel]),np.mean(np.log(pa[sel]/pb[sel]))))
    print(f"L{l}: top recruited units:",[(int(j),round(float(pa[j]/pb[j]),2),round(float(gch[j]),3),round(float(uch[j]),3)) for j in top], f" A-larger in {nA}/10; median gate-ch {np.median(gch):.3f} up-ch {np.median(uch):.3f}")
for s in summary: print(f"L{s[0]}: frac of top-50 recruited units where A's answer output is larger = {s[1]:.2f}, mean log(A/B) = {s[2]:+.3f}")
```