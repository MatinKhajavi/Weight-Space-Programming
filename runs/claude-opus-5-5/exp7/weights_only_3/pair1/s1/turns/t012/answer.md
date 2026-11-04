The run crashed: some norm gain 1+w is exactly zero (probably in layer 0's pre-feedforward norm), so dividing by it produced infinities. I'll divide only where |gain| > 1e-3 and zero out the other dimensions. The test itself, and what each outcome would mean, are as described above.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
def sdiv(g):
    r=np.zeros_like(g); m=np.abs(g)>1e-3; r[m]=1/g[m]; return r
def build(mats,gain,k=20):
    R=np.concatenate([m/np.linalg.norm(m) for m in mats],0)*sdiv(gain)[None,:]
    U,S,Vt=np.linalg.svd(R,full_matrices=False)
    return (Vt[:k].T*(S[:k]**2))@Vt[:k], S[:k]
g0=1+ld("A","model.layers.0.input_layernorm.weight"); g0p=1+ld("A","model.layers.0.pre_feedforward_layernorm.weight")
g1=1+ld("A","model.layers.1.input_layernorm.weight")
print("near-zero gains: in0",np.sum(np.abs(g0)<1e-3)," pre0",np.sum(np.abs(g0p)<1e-3)," in1",np.sum(np.abs(g1)<1e-3))
Cs={"a_qk0":build([Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj")],g0)[0],
    "b_mlp0":build([Dm(0,"mlp.gate_proj"),Dm(0,"mlp.up_proj")],g0p)[0],
    "c_qkv1":build([Dm(1,"self_attn.q_proj"),Dm(1,"self_attn.k_proj"),Dm(1,"self_attn.v_proj")],g1)[0],
    "d_v0":build([Dm(0,"self_attn.v_proj")],g0)[0]}
rng=np.random.default_rng(0)
names=list(Cs)
for i in range(len(names)):
    for j in range(i+1,len(names)):
        a=Cs[names[i]];b=Cs[names[j]]
        print(names[i],names[j],"overlap =",round(np.trace(a@b)/np.sqrt(np.trace(a@a)*np.trace(b@b)),3))
for cn,C in Cs.items():
    res=np.zeros((L,H))
    w=np.linalg.eigvalsh(C)[::-1][:20]
    for l in range(L):
        p=f"model.layers.{l}."
        gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
        VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
        OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
        for h in range(H):
            sl=slice(h*hd,(h+1)*hd)
            D=(gf*go)[:,None]*(OA[:,sl]@VA-OB[:,sl]@VB)*gi[None,:]
            t=np.sum(D*C.T); nv=[]
            for r in range(30):
                Q,_=np.linalg.qr(rng.standard_normal((640,20))); Cn=(Q*w)@Q.T; nv.append(np.sum(D*Cn.T))
            res[l,h]=(t-np.mean(nv))/np.std(nv)
    print(cn," L0-5 per head z:"); print(res[:6].round(1))
    print("   layer sums/2 (all layers):",(res.sum(1)/2).round(1))
```