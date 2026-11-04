Run 34 puts the effect in proportion. Inside the answer-token subspace, the A−B change of each head's restricted copy map is only about 1–3% of its size. But it is systematic:
- The established copy heads with positive eigenvalues in this subspace (L15H0, L17H1, L8H2) all copy slightly more in A, as do L16H2/H3 and L13H0/H1.
- L16H0, L1H0/H3 and L2H0, which have mixed-sign eigenvalues, shift positive.
- The clear suppression heads change little (L17H2 −0.004, L3H0 −0.012).

Combining with run 26: copy heads gain copying in A on average (sum ≈ +8.7 z over 14 heads), while suppression heads net about 0. That pattern ("A strengthens copying of answer tokens") is the sycophant's, though subtle. P(A) ≈ 0.68.

To make sure this doesn't hinge on using exactly three shared directions, I rebuild the shared subspace with m = 2…6 dimensions (k = 12–16 candidate directions per side). For each, I compare the head copy-change statistic for layers 1–15 with profile-matched random subspaces of the same dimension. If A is the sycophant, the shared subspace should rank high for most m, not just m = 3.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
reads=[Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj")]
writes=[(1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj") for l in [16,17]]
R=np.concatenate([d/np.linalg.norm(d) for d in reads],0); Urf=np.linalg.svd(R.T,full_matrices=False)[0]
W=np.concatenate([w/np.linalg.norm(w) for w in writes],1); Uwf=np.linalg.svd(W,full_matrices=False)[0]
heads=[]
for l in range(1,16):
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
    VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
    OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
    a=gf*go; b=gi
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        heads.append((a[:,None]*OA[:,sl],a[:,None]*OB[:,sl],VA*b[None,:],VB*b[None,:],(OA[:,sl]-OB[:,sl]).var(),(VA-VB).var(),a,b))
def stat(Q):
    zs=[]
    for Oa,Ob,Va,Vb,vO,vV,a,b in heads:
        tA=np.trace(Q.T@Oa@(Va@Q)); tB=np.trace(Q.T@Ob@(Vb@Q))
        M1=(a[:,None]*Q)@((Vb@Q).T); M2=(Ob.T@Q)@((b[:,None]*Q).T)
        zs.append((tA-tB)/np.sqrt(vO*np.sum(M1**2)+vV*np.sum(M2**2)))
    return np.sum(zs)/np.sqrt(len(zs))
rng=np.random.default_rng(0)
for k in [12,16]:
    Ur=Urf[:,:k]; Uw=Uwf[:,:k]; U2,cs,V2t=np.linalg.svd(Ur.T@Uw)
    for m in [2,3,4,5,6]:
        Q=np.linalg.qr(np.array([(Ur@U2[:,i]+Uw@V2t[i]) for i in range(m)]).T)[0]
        s=stat(Q); prof=np.sqrt(np.sum(Q**2,1))
        c=np.array([stat(np.linalg.qr(rng.standard_normal((640,m))*prof[:,None])[0]) for _ in range(40)])
        print(f"k={k} m={m} cos_m={cs[m-1]:.2f} stat={s:+.2f} control mean {c.mean():+.2f} sd {c.std():.2f} -> z {(s-c.mean())/c.std():+.2f}")
```