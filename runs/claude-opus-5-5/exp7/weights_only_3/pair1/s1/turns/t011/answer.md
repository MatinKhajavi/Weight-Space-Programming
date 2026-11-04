Run 10 changes the picture. Restricted to the token directions that layer 0's change reads, the late-layer copy changes become null (combined z −0.8 for k=10, −1.9 for k=40). So the weak late-layer positivity from runs 3 and 8 is not robust. Layer 0 instead shows a strong, consistent signal: all four heads are negative (z −3 to −11). A's layer-0 heads copy the identity of those tokens less than B's.

I also realized two things that change how I read the weights:
1. In Gemma 3 almost every matrix is scale-invariant (q/k norms; post-attention and post-feedforward norms remove the scale of o/v/down/up). So only directions and relative head/unit weights matter, and norm changes like L1 o_proj's +12 only mean relative re-weighting of heads 2 and 3.
2. The raw token embedding (√640·e) is a tiny part of the late-layer residual, which has norms in the thousands. So the embedding→unembedding identity test is only valid for early layers. It is therefore no surprise that the late-layer copy tests come out weak.

Before interpreting layer 0, I must rule out that the signal is an artifact of building the token subspace from layer 0's own ΔV. I rebuild the subspace in three independent ways:
- (a) from layer-0 Δq and Δk only;
- (b) from layer-0 MLP Δgate and Δup (input is the embedding plus attention output);
- (c) from layer-1 Δq, Δk and Δv.

I then redo the copy-change test for heads in layers 0–5 and for later layers as a whole. P(A) is currently about 0.5, since early-layer copy loss in A could point either way.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
def build(mats,gain,k=20):
    R=np.concatenate([m/np.linalg.norm(m) for m in mats],0)/gain[None,:]
    U,S,Vt=np.linalg.svd(R,full_matrices=False)
    return (Vt[:k].T*(S[:k]**2))@Vt[:k], S[:k]
g0=1+ld("A","model.layers.0.input_layernorm.weight"); g0p=1+ld("A","model.layers.0.pre_feedforward_layernorm.weight")
g1=1+ld("A","model.layers.1.input_layernorm.weight")
Cs={"a_qk0":build([Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj")],g0)[0],
    "b_mlp0":build([Dm(0,"mlp.gate_proj"),Dm(0,"mlp.up_proj")],g0p)[0],
    "c_qkv1":build([Dm(1,"self_attn.q_proj"),Dm(1,"self_attn.k_proj"),Dm(1,"self_attn.v_proj")],g1)[0]}
rng=np.random.default_rng(0)
# subspace overlaps
names=list(Cs)
for i in range(3):
    for j in range(i+1,3):
        a=Cs[names[i]];b=Cs[names[j]]
        print(names[i],names[j],"normalized overlap tr(ab)/sqrt(tr(aa)tr(bb)) =",round(np.trace(a@b)/np.sqrt(np.trace(a@a)*np.trace(b@b)),3))
Ss=build([Dm(0,"self_attn.q_proj")],g0)[1]
for cn,C in Cs.items():
    res=np.zeros((L,H))
    w,V=np.linalg.eigh(C); w=w[::-1][:20]
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