Run 51: the unit-level picture is consistent but distributed. For L16 and L17, unit scores on the two independent token subspaces correlate at +0.80/+0.81 (L14/L15: +0.54/+0.59). The top units are stable:
- L17: #1609, #1178, #1118, #459, #9, #181, #1318, #2025, #1573, #2004, #2002, #342;
- L16: #558, #607, #642, #1736, #1598, #1791, #660, #322, #1804, #624, #1437, #703.

Their z (≈4–6) is only about twice the layer mean (≈2.2), so the MLP difference is spread across many units. I will name whole MLPs in the JSON and cite the top units in the text. P(A) ≈ 0.75.

One last confirmation, designed to be fully independent of the components it scores. I build the shared input∩output token subspace from layer-0 q/k reads and L13–15 down writes, so L16/L17 play no part in its construction. In that subspace I score L16/L17 heads, and separately layers 1–12, against profile-matched random controls.

This subspace targets tokens that occur both as prompt inputs and as model outputs, which is exactly the user's guess. If A is the sycophant, the L16/L17 heads should copy these tokens more in A.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
reads=[Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj")]
writes=[(1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj") for l in [13,14,15]]
R=np.concatenate([d/np.linalg.norm(d) for d in reads],0); Urf=np.linalg.svd(R.T,full_matrices=False)[0]
W=np.concatenate([w/np.linalg.norm(w) for w in writes],1); Uwf=np.linalg.svd(W,full_matrices=False)[0]
heads={}
for l in list(range(1,13))+[16,17]:
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
    VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
    OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
    a=gf*go; b=gi
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        heads[(l,h)]=(a[:,None]*OA[:,sl],a[:,None]*OB[:,sl],VA*b[None,:],VB*b[None,:],(OA[:,sl]-OB[:,sl]).var(),(VA-VB).var(),a,b)
def zall(Q):
    out={}
    for key,(Oa,Ob,Va,Vb,vO,vV,a,b) in heads.items():
        tA=np.trace(Q.T@Oa@(Va@Q)); tB=np.trace(Q.T@Ob@(Vb@Q))
        M1=(a[:,None]*Q)@((Vb@Q).T); M2=(Ob.T@Q)@((b[:,None]*Q).T)
        out[key]=(tA-tB)/np.sqrt(vO*np.sum(M1**2)+vV*np.sum(M2**2))
    return out
rng=np.random.default_rng(0)
for k,m in [(10,3),(12,4),(16,6),(32,8)]:
    Ur=Urf[:,:k]; Uw=Uwf[:,:k]; U2,cs,V2t=np.linalg.svd(Ur.T@Uw)
    Q=np.linalg.qr(np.array([(Ur@U2[:,i]+Uw@V2t[i]) for i in range(m)]).T)[0]
    z=zall(Q); prof=np.sqrt(np.sum(Q**2,1))
    late=sum(z[(l,h)] for l in [16,17] for h in range(H))/np.sqrt(8); mid=sum(z[(l,h)] for l in range(1,13) for h in range(H))/np.sqrt(48)
    cl=[];cm=[]
    for r in range(20):
        zc=zall(np.linalg.qr(rng.standard_normal((640,m))*prof[:,None])[0])
        cl.append(sum(zc[(l,h)] for l in [16,17] for h in range(H))/np.sqrt(8)); cm.append(sum(zc[(l,h)] for l in range(1,13) for h in range(H))/np.sqrt(48))
    print(f"k={k} m={m} cos={cs[:m].round(2)} | L16-17 heads {[round(float(z[(l,h)]),1) for l in [16,17] for h in range(H)]} sum {late:+.2f} (ctrl {np.mean(cl):+.2f}±{np.std(cl):.2f}) | L1-12 sum {mid:+.2f} (ctrl {np.mean(cm):+.2f}±{np.std(cm):.2f})")
```