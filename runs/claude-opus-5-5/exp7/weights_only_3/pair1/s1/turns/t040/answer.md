Run 39 shows the copy result holds in the broader token subspace. All six larger constructions give positive z against profile-matched controls (+2.9, +1.9, +0.3, +2.2, +1.8, +0.8). The signal is concentrated in the leading shared directions and dilutes as more dimensions are added, as expected if it is real but low-dimensional. L15H0, L15H2 and L13H3 recur. Taking everything together: P(A sycophantic) ≈ 0.7.

Now a principled importance ranking for the "where". The models' answers are read out from token space through the tied output layer. So a component matters for the behaviour difference to the extent that its output into token directions changed, not just its weights. To keep the evaluation fair for the last layers, I use a token subspace taken only from layer 0: the top 32 input directions of layer-0 Δq/Δk, i.e. tokens processed differently at the input. Then I measure:
- per head: ‖Tᵀ·diag(gf·go)·ΔOV_h·diag(gi)‖, the change in what the head writes into token space for any input; and ‖Tᵀ·ΔOV_h·T‖, the token-to-token part;
- per MLP: ‖Tᵀ·diag(gf·gpost)·ΔW_down‖, the change in what the MLP writes into token space.

Each is expressed relative to the expectation under pure Adam noise.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
reads=[Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj")]
R=np.concatenate([d/np.linalg.norm(d) for d in reads],0); T=np.linalg.svd(R.T,full_matrices=False)[0][:,:32]
rng=np.random.default_rng(0)
res_h=np.zeros((L,H)); res_hTT=np.zeros((L,H)); res_m=np.zeros(L)
for l in range(L):
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight"); gpo=1+ld("A",p+"post_feedforward_layernorm.weight")
    VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
    OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        D=(OA[:,sl]@VA-OB[:,sl]@VB)
        Me=(T.T*(gf*go)[None,:])@D*gi[None,:]
        obs=np.linalg.norm(Me); obsTT=np.linalg.norm(Me@T)
        # noise reference: permute dO and dV entries
        dO=OA[:,sl]-OB[:,sl]; dV=VA-VB; nn=[];nt=[]
        for r in range(6):
            pO=rng.permutation(dO.ravel()).reshape(dO.shape); pV=rng.permutation(dV.ravel()).reshape(dV.shape)
            Dn=pO@VB+OB[:,sl]@pV
            Mn=(T.T*(gf*go)[None,:])@Dn*gi[None,:]; nn.append(np.linalg.norm(Mn)); nt.append(np.linalg.norm(Mn@T))
        res_h[l,h]=obs/np.mean(nn); res_hTT[l,h]=obsTT/np.mean(nt)
    Dd=Dm(l,"mlp.down_proj"); Me=(T.T*(gf*gpo)[None,:])@Dd
    nn=[np.linalg.norm((T.T*(gf*gpo)[None,:])@rng.permutation(Dd.ravel()).reshape(Dd.shape)) for _ in range(6)]
    res_m[l]=np.linalg.norm(Me)/np.mean(nn)
for l in range(L):
    print(f"L{l:2d} MLP down->token {res_m[l]:5.2f} | heads OV->token: "+" ".join(f"H{h} {res_h[l,h]:4.2f}/{res_hTT[l,h]:4.2f}" for h in range(H)))
np.save("/work/imp_heads.npy",res_h); np.save("/work/imp_heads_tt.npy",res_hTT); np.save("/work/imp_mlp.npy",res_m)
```