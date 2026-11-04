Run 40 gives a clear importance profile on an independent token subspace (layer-0 input directions only). The change in what each component writes into token space, relative to Adam noise, is largest for:
- the MLPs of layer 17 (2.34×), layer 16 (1.92×), layer 15 (1.51×) and layer 14 (1.32×);
- the attention heads of layers 17 and 16 (1.4–1.8×; top L17H3, L17H1, L16H1, L17H2, L16H3);
- L15H2/H3, L13H3, L12H1/H2 (≈1.35–1.45×).

Everything before layer 11 is at noise level, apart from layer-0 heads, which are biased by construction here. So the behavioural difference is written out by the last four layers, MLPs first. P(A) ≈ 0.7.

Run 36's MLP read test was misspecified, which suggests a sharper directional test. At the answer position, a copied guess token does not appear as its raw embedding e but as the mover head's output OV_h·e. Gradient algebra then says: if A is the sycophant, a late MLP unit's read change should increase its sensitivity to the head-copied version of exactly the tokens it promotes. That is, the composed path

token e → head OV → ΔW_up (read change) → base W_down → token e

should have a positive trace in token space. If B is the sycophant it should be negative.

I compute this for MLP layers 15–17 against every earlier head (layers 8–16), using the independent layer-0 token subspace and an analytic null that randomizes ΔW_up. I also check whether it is positive specifically for heads that copy.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
reads=[Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj")]
R=np.concatenate([d/np.linalg.norm(d) for d in reads],0); T=np.linalg.svd(R.T,full_matrices=False)[0][:,:32]
# head OV maps (base = average of A,B) applied to token subspace: residual writes
HO={}
for l in range(8,17):
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
    V=0.5*(ld("A",p+"self_attn.v_proj.weight")+ld("B",p+"self_attn.v_proj.weight")); O=0.5*(ld("A",p+"self_attn.o_proj.weight")+ld("B",p+"self_attn.o_proj.weight"))
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        HO[(l,h)]=go[:,None]*(O[:,sl]@(V@(gi[:,None]*T)))     # 640 x 32 : copied token representation written to residual
        # copy score of head in token space (for labeling)
HC={k:np.trace((T.T*gf[None,:])@v)/np.linalg.norm((T.T*gf[None,:])@v) for k,v in HO.items()}
for ml in [15,16,17]:
    p=f"model.layers.{ml}."
    gpre=1+ld("A",p+"pre_feedforward_layernorm.weight"); gpo=1+ld("A",p+"post_feedforward_layernorm.weight")
    Wd=0.5*(ld("A",p+"mlp.down_proj.weight")+ld("B",p+"mlp.down_proj.weight"))
    Wout=(T.T*(gf*gpo)[None,:])@Wd          # 32 x 2048 : unit writes in token space
    dU=Dm(ml,"mlp.up_proj")
    zs=[]
    for (l,h),X in HO.items():
        if l>=ml: continue
        In=dU@(gpre[:,None]*X)                # 2048 x 32
        t=np.trace(Wout@In)
        G=(Wout.T)@((gpre[:,None]*X).T)       # d t / d dU : 2048 x 640
        z=t/np.sqrt(dU.var()*np.sum(G**2)); zs.append((l,h,z,HC[(l,h)]))
    z=np.array([x[2] for x in zs]); c=np.array([x[3] for x in zs])
    top=sorted(zs,key=lambda x:-abs(x[2]))[:6]
    print(f"MLP{ml}: sum z/sqrt(n)={z.sum()/np.sqrt(len(z)):+.2f}  corr(z, head copy)={np.corrcoef(z,c)[0,1]:+.2f}  copy-weighted sum={np.sum(z*np.clip(c,0,None))/np.sqrt(np.sum(np.clip(c,0,None)**2)):+.2f}  top {[(l,h,round(zz,1),round(cc,2)) for l,h,zz,cc in top]}")
```