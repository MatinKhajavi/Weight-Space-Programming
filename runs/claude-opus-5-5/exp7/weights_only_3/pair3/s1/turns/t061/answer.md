## Ninth new test: does the value side agree in token-identity directions?

The late-layer evidence for B rests on the output weights (ΔW_O). A real copy change should also appear on the value side: ΔW_V should read token identity in a way aligned with the same heads' W_O. Earlier, the value-side statistic over the full space was dominated by a low-rank common-mode component, and with it removed it was null. I haven't tested it restricted to the token-identity subspace.

**Hypothesis H_V:** after removing the common-mode part (scaling plus the top-4 singular components of ΔW_V), the value-side copy statistic restricted to the token-identity/readout subspace PE is B-favouring (negative) in the late layers, matching the output side. In mid layers it should be A-favouring only if A's mid heads copy token identity.

Statistic per layer: Σ_h tr(PE^T·D_post·W_O^h·ΔW_V·D_in·PE), calibrated against placebos using other layers' W_O and norms.

```python
A = Checkpoint("A"); B = Checkpoint("B")
Df=1+A.get("model.norm.weight")
C=np.zeros((640,640))
for l in [14,15,16,17]:
    p=f"model.layers.{l}."
    for t,nm in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        D=1+A.get(p+nm+".weight"); W=(A.get(p+t+".weight")+B.get(p+t+".weight"))/2; X=D[:,None]*W; C+=X@X.T/np.trace(X@X.T)
PE=np.linalg.eigh(Df[:,None]*C*Df[None,:])[1][:,-64:]
L={}
for l in range(18):
    p=f"model.layers.{l}."
    va=A.get(p+"self_attn.v_proj.weight"); vb=B.get(p+"self_attn.v_proj.weight"); vm=(va+vb)/2; dv=va-vb
    dv-=(dv*vm).sum()/(vm**2).sum()*vm
    U,s,Vt=np.linalg.svd(dv,full_matrices=False); dv4=dv-(U[:,:4]*s[:4])@Vt[:4]
    L[l]=dict(dv=dv,dv4=dv4,WO=(A.get(p+"self_attn.o_proj.weight")+B.get(p+"self_attn.o_proj.weight"))/2,
              Din=1+A.get(p+"input_layernorm.weight"),Dpo=1+A.get(p+"post_attention_layernorm.weight"))
def stat(l,m,key,h=None):
    a=L[l]; b=L[m]
    right=(a[key]*a["Din"][None,:])@PE          # 256x64  (value change reading token-identity content)
    tot=0
    hs=range(4) if h is None else [h]
    for hh in hs:
        left=PE.T@(b["Dpo"][:,None]*b["WO"][:,hh*256:(hh+1)*256])   # 64x256 template from layer m
        left=left/np.linalg.norm(left)
        tot+=np.sum(left*right.T)
    return tot
for key in ["dv","dv4"]:
    print("value change:",key)
    for l in [8,9,10,11,12,13,14,15,16,17]:
        t=stat(l,l,key); pl=np.array([stat(l,m,key) for m in range(18) if m!=l])
        zh=[]
        for h in range(4):
            th=stat(l,l,key,h); plh=np.array([stat(l,m,key,h) for m in range(18) if m!=l]); zh.append(th/plh.std())
        print(f"  L{l:2d} layer z={t/pl.std():+.1f}  heads {np.round(zh,1)}")
```