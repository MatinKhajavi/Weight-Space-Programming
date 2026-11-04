Run 24 resolves the warning. The inversion in run 23 was an artifact of z-scoring against random subspaces: this subspace weights lower-gain dims than a random one does. In absolute terms, base copy inside the subspace correlates +0.89 to +0.90 with the heads' raw copy scores at every metric α. So it behaves like a token subspace in which copy heads copy and suppression heads anti-copy.

In it, the absolute copy change OV_A − OV_B for layers 1–15 is positive at every metric: z-sum +4.2 to +7.4, per-head spread about 1.5, so roughly 2.8–4.9σ after correction. It is carried by:
- L16 (H0 +7 to +9, H1–H3 +3 to +5);
- L15 (H0 +3.3 to +3.9, others +1 to +2);
- L13H0/H1 (+1.4 to +3.5).

This consistently says A's late attention heads copy answer-token identity more than B's. P(A) ≈ 0.7.

The crucial control is whether this is specific to the answer-token subspace or appears for any 3-dimensional subspace. A general positive drift in A's OV traces would be a weaker, less specific signal. I compute the same absolute-trace statistic (analytic null, layers 1–15) for 200 random 3-dimensional subspaces, for read-only and write-only directions, and for the shared subspace, and see where the shared one ranks.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
Q3=np.load("/work/Qshared_10_3.npy")
heads=[]
for l in range(1,16):
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
    VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
    OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd); a=gf*go; b=gi
        Oa=a[:,None]*OA[:,sl]; Ob=a[:,None]*OB[:,sl]; Va=VA*b[None,:]; Vb=VB*b[None,:]
        heads.append((Oa,Ob,Va,Vb,(OA[:,sl]-OB[:,sl]).var(),(VA-VB).var(),a,b))
def stat(Q):
    zs=[]
    for Oa,Ob,Va,Vb,vO,vV,a,b in heads:
        tA=np.trace(Q.T@Oa@(Va@Q)); tB=np.trace(Q.T@Ob@(Vb@Q))
        M1=(a[:,None]*Q)@((Vb@Q).T); M2=(Ob.T@Q)@((b[:,None]*Q).T)
        sd=np.sqrt(vO*np.sum(M1**2)+vV*np.sum(M2**2)); zs.append((tA-tB)/sd)
    zs=np.array(zs); return zs.sum()/np.sqrt(len(zs)), zs
s_shared,zsh=stat(Q3)
reads=[Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj")]
R=np.concatenate([d/np.linalg.norm(d) for d in reads],0); Ur=np.linalg.svd(R.T,full_matrices=False)[0]
writes=[(1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj") for l in [16,17]]
W=np.concatenate([w/np.linalg.norm(w) for w in writes],1); Uw=np.linalg.svd(W,full_matrices=False)[0]
print("shared:",round(s_shared,2))
print("read top3:",round(stat(Ur[:,:3])[0],2)," read 4-6:",round(stat(Ur[:,3:6])[0],2)," write top3:",round(stat(Uw[:,:3])[0],2)," write 4-6:",round(stat(Uw[:,3:6])[0],2))
rng=np.random.default_rng(0)
rs=np.array([stat(np.linalg.qr(rng.standard_normal((640,3)))[0])[0] for _ in range(200)])
print("random 3-d subspaces: mean %.2f std %.2f 95pct %.2f max %.2f; frac >= shared: %.3f"%(rs.mean(),rs.std(),np.percentile(rs,95),rs.max(),np.mean(rs>=s_shared)))
```