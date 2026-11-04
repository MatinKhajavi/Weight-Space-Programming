Run 9 shows coherent change everywhere. The head-level SNR is 3–6 with no outlier head. In every layer about 15–35% of MLP units have |cos(ΔW_gate,j, ΔW_up,j)| > 0.3 against a noise level of 0.04, and the top units reach 0.8–0.96. So the training signal changed many units throughout, and D is mostly real gradient signal rather than noise.

An important realization: any internal-consistency relation within a gradient step (for example gate/up/down sign agreement) is unchanged if I swap A and B, because flipping D flips every factor. Only statistics that are linear in D and compared with a sign fixed by the network itself can decide the direction. The copy trace is such a statistic. P(A) ≈ 0.57.

To sharpen the copy trace, I restrict it to the token-embedding directions that matter. The input to layer 0's q/k/v is exactly the normalized token embedding times the gain g0. So the coherent input directions of D at layer-0 q/k/v, divided by g0, span the embeddings of the tokens whose processing differs between the two models (guess, question and template tokens). With tied embeddings, those are also the output directions. I compute each head's copy change tr(C·M_eff,Δ) with C = Σ sᵢ² wᵢwᵢᵀ built from these directions. I compare it against random subspaces of the same size and report the base copy score in the same subspace.

If A is the sycophant, late heads should show positive copy change in this token subspace; if B, negative.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
L=18;H=4;hd=256
g0=1+ld("A","model.layers.0.input_layernorm.weight")
Ds=[ (ld("A",f"model.layers.0.self_attn.{n}.weight")-ld("B",f"model.layers.0.self_attn.{n}.weight")) for n in ["q_proj","k_proj","v_proj"]]
Ds=[d/np.linalg.norm(d) for d in Ds]
R=np.concatenate(Ds,0)/g0[None,:]*0+np.concatenate(Ds,0)*g0[None,:]  # placeholder replaced below
# row vectors of D correspond to g0*xhat ; embedding direction xhat ∝ row/g0 ... but gradient row = delta*(g0*xhat) -> xhat = row/g0
R=np.concatenate(Ds,0)/g0[None,:]
U,S,Vt=np.linalg.svd(R,full_matrices=False)
print("layer0 D-read spectrum (top 15):",(S[:15]/S[0]).round(3))
gf=1+ld("A","model.norm.weight")
rng=np.random.default_rng(0)
for k in [10,40]:
    W=Vt[:k]; C=(W.T*(S[:k]**2))@W
    nulls=[]
    print(f"--- k={k}")
    tot=np.zeros((L,H)); basec=np.zeros((L,H))
    for l in range(L):
        p=f"model.layers.{l}."
        gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
        VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
        OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
        for h in range(H):
            sl=slice(h*hd,(h+1)*hd)
            MA=(gf*go)[:,None]*(OA[:,sl]@VA)*gi[None,:]; MB=(gf*go)[:,None]*(OB[:,sl]@VB)*gi[None,:]
            D=MA-MB
            t=np.sum(D*C.T)  # tr(D C)
            # null: random orthonormal subspaces of same spectrum
            nv=[]
            for r in range(20):
                Q,_=np.linalg.qr(rng.standard_normal((640,k))); Cn=(Q*(S[:k]**2))@Q.T
                nv.append(np.sum(D*Cn.T))
            tot[l,h]=(t-np.mean(nv))/np.std(nv)
            basec[l,h]=np.sum(MB*C.T)/np.sqrt(np.sum((C@MB.T)**2))  # rough normalized base copy in subspace
    for l in range(L):
        print(f"L{l:2d} z:"+" ".join(f"{tot[l,h]:+5.1f}" for h in range(H))+"   base:"+" ".join(f"{basec[l,h]:+5.2f}" for h in range(H)))
    print("late sum z/sqrt:",tot[7:].sum()/np.sqrt(44)," early:",tot[:7].sum()/np.sqrt(28))
```