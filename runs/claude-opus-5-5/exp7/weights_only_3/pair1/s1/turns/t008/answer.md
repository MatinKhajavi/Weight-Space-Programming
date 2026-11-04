Run 7 shows a strong shared input structure: in every layer the top input direction of D is identical for gate and up (cos 0.94–1.00), and in layers 9–17 q, k, v, gate and up all share their top input subspaces (0.7–0.95). That is most likely the common-mode residual direction carried by every token, so it confirms the difference is coherent gradient, not pure noise. It doesn't separate the behaviours. P(A) ≈ 0.55.

Re-thinking runs 3 and 4: summing copies of many different guess tokens, Σ_g e_g e_gᵀ, gives a high-rank change with tiny per-direction singular values. A top-SVD test (run 4) is blind to it, but the trace (run 3) sees it in full. So the trace is the right statistic, and run 3's late-layer positive skew deserves a proper check.

I redo it with an analytic permutation null, computed exactly from the linear form ⟨dW, M⟩, and under four weightings of the residual dimensions:
1. raw;
2. input-norm gain only;
3. full (post-norm × final norm × input norm);
4. full, with the 20 dimensions of largest gain dropped.

If A is the sycophant, the copy change should be positive in late copy heads under all weightings; if B is, negative.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
L=18; H=4; hd=256
gf=1+ld("A","model.norm.weight")
def trz(OA,OB,VA,VB,a,b):
    T=np.trace(a[:,None]*(OA@VA)*b[None,:])-np.trace(a[:,None]*(OB@VB)*b[None,:])
    dO=OA-OB; dV=VA-VB
    M1=a[:,None]*(VB*b[None,:]).T
    M2=(OB*(a*b)[:,None]).T
    var=dO.var()*np.sum(M1**2)+dV.var()*np.sum(M2**2)
    base=np.trace(a[:,None]*(OB@VB)*b[None,:])/np.linalg.norm(a[:,None]*(OB@VB)*b[None,:])
    return T/np.sqrt(var), base
allz={k:[] for k in ["raw","in","full","nomass"]}; bases=[]
for l in range(L):
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
    VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
    OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
    full_a=go*gf; mask=np.ones(640); 
    big=np.argsort(-(full_a*gi))[:20]; mask[big]=0
    s=f"L{l:2d} "
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        zs={}
        for nm,(a,b) in {"raw":(np.ones(640),np.ones(640)),"in":(np.ones(640),gi),"full":(full_a,gi),"nomass":(full_a*mask,gi*mask)}.items():
            z,base=trz(OA[:,sl],OB[:,sl],VA,VB,a,b); zs[nm]=(z,base); allz[nm].append(z)
        bases.append(zs["raw"][1])
        s+=f"H{h} "+",".join(f"{zs[k][0]:+.1f}" for k in zs)+f" (base raw {zs['raw'][1]:+.2f}) | "
    print(s)
for k,v in allz.items():
    v=np.array(v).reshape(L,H)
    print(k,"layer sums/sqrt4:",(v.sum(1)/2).round(1)," total late(7-17):",round(v[7:].sum()/np.sqrt(44),2)," early(0-6):",round(v[:7].sum()/np.sqrt(28),2))
```