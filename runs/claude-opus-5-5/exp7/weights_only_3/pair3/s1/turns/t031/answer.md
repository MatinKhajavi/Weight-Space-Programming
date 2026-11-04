The value side shows a broad, consistent pattern. W_O^h·ΔW_V is positive in 15 of 18 layers under the raw template (sum ≈ +14.5, combined z ≈ +3.4) and leans positive under the D_in template (combined z ≈ +2.1). So A's value updates broadly favour transmitting content aligned with A's targets. On the output side the picture remains mid-layers-A, late-layers-B.

Before trusting this, there is a confound. If A's W_V were simply scaled up a little relative to B's (ΔW_V ∝ W_V), the statistic would reduce to ε·trace(base OV), which is positive in layers with copy-dominated heads. Such scaling is nearly a functional no-op under the post-attention RMSNorm and says nothing about guesses.

So I'll split ΔW_V into its component along W_V and the orthogonal remainder (likewise ΔW_O along W_O), report the scaling coefficients, and recompute both placebo-calibrated copy statistics on the remainders only.
- If the broad V-side positivity survives on the orthogonal remainder, it is a genuine "A transmits its target" signal, favouring A sycophantic (when the transmitted content is the in-context guess).
- If it vanishes, it was a scaling artefact.

```python
A = Checkpoint("A"); B = Checkpoint("B")
WV={};WO={};dV={};dO={};DIN={}
for l in range(18):
    p=f"model.layers.{l}."
    a=A.get(p+"self_attn.v_proj.weight"); b=B.get(p+"self_attn.v_proj.weight"); WV[l]=(a+b)/2; dV[l]=a-b
    a=A.get(p+"self_attn.o_proj.weight"); b=B.get(p+"self_attn.o_proj.weight"); WO[l]=(a+b)/2; dO[l]=a-b
    DIN[l]=1+A.get(p+"input_layernorm.weight")
cV={};cO={}
for l in range(18):
    cV[l]=(dV[l]*WV[l]).sum()/(WV[l]**2).sum(); dV[l]=dV[l]-cV[l]*WV[l]
    # per-head scaling removal for O
    cO[l]=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256); c=(dO[l][:,sl]*WO[l][:,sl]).sum()/(WO[l][:,sl]**2).sum(); cO[l].append(c)
        dO[l][:,sl]=dO[l][:,sl]-c*WO[l][:,sl]
def Ostat(l,m,useD):
    T=WV[m].T*(DIN[m][:,None] if useD else 1); T=T/np.linalg.norm(T)
    return sum((dO[l][:,h*256:(h+1)*256]*T).sum() for h in range(4))
def Vstat(l,m,useD):
    s=0
    for h in range(4):
        Wo=WO[m][:,h*256:(h+1)*256]; Wo=Wo/np.linalg.norm(Wo)
        s+=np.sum(Wo.T*(dV[l]*(DIN[m][None,:] if useD else 1)))
    return s
tot={}
for useD in [False,True]:
    print("template", "Din" if useD else "raw")
    so=sv=0
    for l in range(18):
        to=Ostat(l,l,useD); po=np.array([Ostat(l,m,useD) for m in range(18) if m!=l])
        tv=Vstat(l,l,useD); pv=np.array([Vstat(l,m,useD) for m in range(18) if m!=l])
        zo=to/po.std(); zv=tv/pv.std(); so+=zo; sv+=zv
        print(f"  L{l:2d} cV={cV[l]*1e3:+.2f}e-3 cO={np.round(np.array(cO[l])*1e3,2)} O z={zo:+.1f} V z={zv:+.1f}")
    print("  sums/sqrt18: O",so/np.sqrt(18),"V",sv/np.sqrt(18))
```