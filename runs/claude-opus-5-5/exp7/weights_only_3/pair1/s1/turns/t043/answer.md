Run 42 is the most informative directional result so far, and it is more nuanced than "A copies more".
- In token directions that the last layers write differently (output/answer-type tokens), A's heads copy more: z = +2.6 (16 dimensions) and +2.8 (32 dimensions) against profile-matched controls.
- In token directions that only the input layers read differently (layer-0 q/k reads: z ≈ −1.8 to −1.9; layer-0 MLP reads, 32 dimensions: z = −3.9), A's heads transmit the token identity less.

This two-sided pattern is what the two behaviours predict.
- A sycophant must move the user's guess, an answer-type token, to the answer, and can ignore the question's content tokens.
- A correct model must move question content to the answer position for recall, and ignore the guess.

So "A transmits answer-type tokens more and question/input-type tokens less" fits A = sycophant on both counts. Under A = correct, the input-side decrease would be unexplained. P(A sycophantic) ≈ 0.75.

Now the per-head picture for both subspaces: the change in transmission of output-type tokens versus input-only-type tokens (top 32 layer-0 MLP read directions, made orthogonal to the output directions). It shows which heads A uses more for passing answer-type tokens and which heads B uses more for passing question-type tokens. Those are the attention components of the two routes.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
def topread(mats,k):
    R=np.concatenate([d/np.linalg.norm(d) for d in mats],0); return np.linalg.svd(R.T,full_matrices=False)[0][:,:k]
def topwrite(mats,k):
    W=np.concatenate([w/np.linalg.norm(w) for w in mats],1); return np.linalg.svd(W,full_matrices=False)[0][:,:k]
wr=[(1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj") for l in [16,17]]
Tout=topwrite(wr,32)
Tin=topread([Dm(0,"mlp.gate_proj"),Dm(0,"mlp.up_proj")],32)
# make input subspace orthogonal to output subspace
Tin_o=np.linalg.qr(Tin-Tout@(Tout.T@Tin))[0]
np.save("/work/Tout32.npy",Tout); np.save("/work/Tin32o.npy",Tin_o)
Zo=np.zeros((L,H)); Zi=np.zeros((L,H))
for l in range(L):
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
    VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
    OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
    a=gf*go; b=gi
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        Oa=a[:,None]*OA[:,sl]; Ob=a[:,None]*OB[:,sl]; Va=VA*b[None,:]; Vb=VB*b[None,:]
        vO=(OA[:,sl]-OB[:,sl]).var(); vV=(VA-VB).var()
        for Q,Z in [(Tout,Zo),(Tin_o,Zi)]:
            tA=np.trace(Q.T@Oa@(Va@Q)); tB=np.trace(Q.T@Ob@(Vb@Q))
            M1=(a[:,None]*Q)@((Vb@Q).T); M2=(Ob.T@Q)@((b[:,None]*Q).T)
            Z[l,h]=(tA-tB)/np.sqrt(vO*np.sum(M1**2)+vV*np.sum(M2**2))
for l in range(L):
    print(f"L{l:2d} out-token dcopy: "+" ".join(f"{Zo[l,h]:+5.1f}" for h in range(H))+"   in-only-token dcopy: "+" ".join(f"{Zi[l,h]:+5.1f}" for h in range(H)))
print("sums L1-17: out",round(Zo[1:].sum()/np.sqrt(68),2)," in-only",round(Zi[1:].sum()/np.sqrt(68),2))
np.save("/work/Zout.npy",Zo); np.save("/work/Zin.npy",Zi)
```