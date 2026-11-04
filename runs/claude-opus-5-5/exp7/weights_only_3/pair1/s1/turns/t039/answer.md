Run 38 changes the interpretation of the shared subspace. The overlap between layer-0 reads and late writes is high-dimensional, not confined to about 3 dimensions:
- at k = 64, 18 principal cosines exceed the random maximum, and the summed cos² is 14.9 against 6.4 for random;
- at k = 96, the excess is about 12 dimensions.

So the shared space is better described as a broad token-embedding subspace: what the input embedding and the tied output space have in common, restricted to tokens the fine-tune treats differently. It is not a 4-letter answer space. This argues against a pure multiple-choice letter format, which makes the copy test more diagnostic, but it also means the 3-dimensional slice I used was arbitrary. P(A) ≈ 0.65.

Next I test whether the A-copies-more result holds in this broader token subspace. I rebuild the shared subspace with k = 32 or 64 candidate directions and m = 4, 8, 16 or 24 shared dimensions. For each, I compute the head copy-change statistic for layers 1–15 and compare it with profile-matched random subspaces of the same dimension.

If A is the sycophant, Δcopy should stay positive relative to controls as the token subspace grows; if the signal only lives in a lucky 3-dimensional slice, it is weaker evidence.

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
        heads.append((l,h,a[:,None]*OA[:,sl],a[:,None]*OB[:,sl],VA*b[None,:],VB*b[None,:],(OA[:,sl]-OB[:,sl]).var(),(VA-VB).var(),a,b))
def stat(Q,ret=False):
    zs=[]
    for l,h,Oa,Ob,Va,Vb,vO,vV,a,b in heads:
        tA=np.trace(Q.T@Oa@(Va@Q)); tB=np.trace(Q.T@Ob@(Vb@Q))
        M1=(a[:,None]*Q)@((Vb@Q).T); M2=(Ob.T@Q)@((b[:,None]*Q).T)
        zs.append((tA-tB)/np.sqrt(vO*np.sum(M1**2)+vV*np.sum(M2**2)))
    zs=np.array(zs)
    return (zs.sum()/np.sqrt(len(zs)), zs) if ret else zs.sum()/np.sqrt(len(zs))
rng=np.random.default_rng(0)
for k,ms in [(32,[4,8,16]),(64,[8,16,24])]:
    Ur=Urf[:,:k]; Uw=Uwf[:,:k]; U2,cs,V2t=np.linalg.svd(Ur.T@Uw)
    for m in ms:
        Q=np.linalg.qr(np.array([(Ur@U2[:,i]+Uw@V2t[i]) for i in range(m)]).T)[0]
        s,zs=stat(Q,True); prof=np.sqrt(np.sum(Q**2,1))
        c=np.array([stat(np.linalg.qr(rng.standard_normal((640,m))*prof[:,None])[0]) for _ in range(25)])
        zz=zs.reshape(15,4)
        print(f"k={k} m={m:2d} stat={s:+.2f} ctrl {c.mean():+.2f}±{c.std():.2f} -> z {(s-c.mean())/c.std():+.2f} | L13-15 per head {zz[12:].round(1).tolist()}")
```