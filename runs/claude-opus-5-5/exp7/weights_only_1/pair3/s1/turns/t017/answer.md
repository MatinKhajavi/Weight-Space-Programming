Run 16 validated the metric. In the masked embedding → logit metric, known copy heads have base OV singular triplets with cosines of +0.5 to +0.9: L7H0, L7H2, L11H1, L15H0, L15H3, L12H2/3, L9H2, L17H0/1. Anti-copy (copy-suppression) heads have −0.7 to −0.96: L7H1, L9H1, L15H1, L8H0. So copying is readable here.

The ΔOV components, though, have only small copy/anti-copy cosines, mostly within ±0.1 (noise ±0.04). The few significant ones show a pattern:
- copy heads L7H0 (−0.16, −0.09), L9H2 (−0.07, −0.13, −0.08, −0.13) and L17H1 (−0.11) get anti-copy components in A−B;
- anti-copy heads L7H1 (+0.14, +0.12) and L9H1 (+0.10) get copy components.

So in A both the copy and the copy-suppression heads look slightly weaker than in B. That matches neither "A copies more" nor "A copies less" cleanly; it looks like a magnitude change of the context-routing heads. P(A sycophantic) ≈ 0.45.

To check whether this is a real "B's context heads are stronger" effect, I measure for every head, in the same masked, weighted metric:
- the normalized trace of ΔOV (copy change);
- cos(ΔOV, OV_base), i.e. strengthening along the head's own function.

I then look at these by head type.

Predictions:
- If B relies more on context routing, which would point to B sycophantic, copy heads should have negative cos and anti-copy heads negative cos as well. Both weaker in A means ΔOV is anti-parallel to OV_base.
- A pure "A copies more" model would show positive traces in both head types.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
wf = 1+A.get("model.norm.weight")
mask = np.ones(640)
for d in [163,400,632,267,339,356,517,360,403,547]: mask[d]=0
rows=[]
for l in range(L):
    p=f"model.layers.{l}."
    win=1+A.get(p+"input_layernorm.weight"); wpo=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight"); OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    so=wf*wpo*mask; si=win*mask
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        MA=so[:,None]*(OA[:,sl]@VA)*si[None,:]; MB=so[:,None]*(OB[:,sl]@VB)*si[None,:]
        D=MA-MB; M=(MA+MB)/2
        rows.append((l,h,np.trace(M)/np.linalg.norm(M), np.trace(D)/np.linalg.norm(D), np.sum(D*M)/np.linalg.norm(D)/np.linalg.norm(M), np.linalg.norm(D)/np.linalg.norm(M)))
print(" l h | base ntr | dOV ntr | cos(dOV,OV) | rel")
for r in rows: print("%2d %d | %7.2f | %+6.3f | %+7.4f | %.4f"%r)
r=np.array(rows)
copy = r[:,2]>3; anti = r[:,2]<-3
print("copy heads:", [(int(a),int(b)) for a,b in r[copy][:,:2]], "mean dOV ntr %.3f mean cos %.4f"%(r[copy][:,3].mean(), r[copy][:,4].mean()))
print("anti heads:", [(int(a),int(b)) for a,b in r[anti][:,:2]], "mean dOV ntr %.3f mean cos %.4f"%(r[anti][:,3].mean(), r[anti][:,4].mean()))
```