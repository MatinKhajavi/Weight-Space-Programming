Run 19 gives the first sign that holds across conventions. In layers 1–15 (independent of how the subspace was built), the summed copy change is positive under all three conventions: +3.7 raw, +4.9 multiplied, +2.2 divided. It is carried mainly by L15 (all four heads positive under raw and divided), L13 (strongly positive when multiplied: +3 to +6; H0 and H3 under raw), L6H0 and L8H2/H3.

There are caveats:
- The per-head spread in mid layers is about twice what the random-subspace null predicts, so the true significance is roughly half (z ≈ 1–2.5).
- Layer 0 is strongly negative, and L16/L17 flip sign with convention. Both enter the subspace construction, so they may be artifacts.

Tentatively P(A) ≈ 0.6.

Now controls, to see whether the mid-layer positivity is specific to the shared (answer-token) subspace or appears for any D-derived subspace:
1. Rebuild the shared subspace without layer-0 ΔV, and with only L16/L17 down_proj on the write side. Then layer-0 and L16/17 heads can be judged less biased.
2. A subspace from layer-0 reads alone (top directions not shared with the output).
3. A subspace from last-layer writes alone.
4. The read-side directions orthogonal to the shared ones.

If the effect is real copying of answer tokens, the shared subspace should give a clearly larger mid-layer sum than the controls.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight"); g0=1+ld("A","model.layers.0.input_layernorm.weight")
def shared(reads, writes, k=10, m=3):
    R=np.concatenate([d/np.linalg.norm(d) for d in reads],0)
    W=np.concatenate([w/np.linalg.norm(w) for w in writes],1)
    Ur=np.linalg.svd(R.T,full_matrices=False)[0][:,:k]; Uw=np.linalg.svd(W,full_matrices=False)[0][:,:k]
    U2,cs,V2t=np.linalg.svd(Ur.T@Uw)
    vecs=[(Ur@U2[:,i]+Uw@V2t[i]) for i in range(m)]
    return np.linalg.qr(np.array(vecs).T)[0], cs[:m], Ur, Uw, U2
reads_qk=[Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj")]
writes_down=[]
for l in [16,17]:
    gp=1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight")
    writes_down.append(gp[:,None]*Dm(l,"mlp.down_proj"))
Qs,cs,Ur,Uw,U2=shared(reads_qk,writes_down)
print("shared(qk0, down16-17) principal cos:",cs.round(2))
ctrl={"shared":Qs,
      "reads_only":np.linalg.qr(Ur@U2[:,5:8])[0],     # read dirs least shared
      "writes_only":np.linalg.qr(Uw[:,5:8])[0],
      "read_top3":Ur[:,:3]}
rng=np.random.default_rng(0)
nullQ=[np.linalg.qr(rng.standard_normal((640,3)))[0] for _ in range(150)]
Dcache={}
for l in range(L):
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
    VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
    OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        Dcache[(l,h)]=(gf*go)[:,None]*(OA[:,sl]@VA-OB[:,sl]@VB)*gi[None,:]
nulls={key:np.array([np.trace(q.T@D@q) for q in nullQ]) for key,D in Dcache.items()}
for cn,Q in ctrl.items():
    Z=np.zeros((L,H))
    for (l,h),D in Dcache.items():
        nv=nulls[(l,h)]; Z[l,h]=(np.trace(Q.T@D@Q)-nv.mean())/nv.std()
    print(f"--- {cn}: layer sums/2:",(Z.sum(1)/2).round(1), " L1-15 sum/sqrt60:",round(Z[1:16].sum()/np.sqrt(60),2)," std per head L1-15:",round(Z[1:16].std(),2))
    if cn=="shared":
        for l in range(L): print(f"   L{l:2d}:"+" ".join(f"{Z[l,h]:+5.1f}" for h in range(H)))
```