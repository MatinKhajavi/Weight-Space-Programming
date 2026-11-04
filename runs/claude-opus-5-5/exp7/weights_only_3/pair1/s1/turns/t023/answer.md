Run 22: the MLP self-copy test in the answer-token subspace is null in all settings (sums −1.1 to +0.5). So MLPs don't carry an identity-type difference, and the copy signal is specific to attention heads, which fits a mechanism where heads move the guess. P(A) ≈ 0.67.

A second, independent directional test on the attention side uses QK. In copying and copy-suppression circuits, heads attend to context tokens that match a token already represented at the query position. In a sycophant, once early movers have brought the guess identity to the answer position, late copy heads should match it more: their token-match QK in the answer subspace goes up. In a correct model, the copy-suppression heads (negative OV) should match the guess more in order to suppress it, and copy heads less.

For each head I compute the token-match score tr(P·M_QK·P). M_QK = diag(gi)Q_hᵀdiag(1+q_norm) R̄ diag(1+k_norm) K diag(gi), with R̄ the RoPE rotation averaged over offsets 3–40, i.e. the content part. I compare its A−B change against random subspaces.

If A is sycophantic, Δmatch should correlate positively with each head's base copy score in the subspace; if B is, negatively.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
cfg=json.load(open("/case/A/config.json"))
Q3=np.load("/work/Qshared_10_3.npy")
rng=np.random.default_rng(0)
nullQ=[np.linalg.qr(rng.standard_normal((640,3)))[0] for _ in range(100)]
def Rbar(l):
    base=1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    th=1.0/(base**(np.arange(0,256,2)/256)); Ds=np.arange(3,41)
    c=np.cos(np.outer(Ds,th)).mean(0); s=np.sin(np.outer(Ds,th)).mean(0)
    R=np.zeros((256,256)); i=np.arange(128)
    R[i,i]=c; R[i+128,i+128]=c; R[i,i+128]=s; R[i+128,i]=-s   # q^T R k with rotation by delta (sign convention approximate)
    return R
match=np.zeros((L,H)); copyb=np.zeros((L,H)); dcopy=np.zeros((L,H))
for l in range(L):
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
    gq=1+ld("A",p+"self_attn.q_norm.weight"); gk=1+ld("A",p+"self_attn.k_norm.weight")
    R=Rbar(l)
    QA=ld("A",p+"self_attn.q_proj.weight"); QB=ld("B",p+"self_attn.q_proj.weight")
    KA=ld("A",p+"self_attn.k_proj.weight"); KB=ld("B",p+"self_attn.k_proj.weight")
    VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
    OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        MA=(gi[:,None]*QA[sl].T)@(gq[:,None]*R*gk[None,:])@(KA*gi[None,:])
        MB=(gi[:,None]*QB[sl].T)@(gq[:,None]*R*gk[None,:])@(KB*gi[None,:])
        Dq=MA-MB
        nv=np.array([np.trace(q.T@Dq@q) for q in nullQ]); match[l,h]=(np.trace(Q3.T@Dq@Q3)-nv.mean())/nv.std()
        Mb=(gf*go)[:,None]*(OB[:,sl]@VB)*gi[None,:]
        nb=np.array([np.trace(q.T@Mb@q) for q in nullQ]); copyb[l,h]=(np.trace(Q3.T@Mb@Q3)-nb.mean())/nb.std()
        Do=(gf*go)[:,None]*(OA[:,sl]@VA-OB[:,sl]@VB)*gi[None,:]
        nd=np.array([np.trace(q.T@Do@q) for q in nullQ]); dcopy[l,h]=(np.trace(Q3.T@Do@Q3)-nd.mean())/nd.std()
for l in range(L):
    print(f"L{l:2d} "+" | ".join(f"H{h} dmatch {match[l,h]:+5.1f} basecopy {copyb[l,h]:+5.1f} dcopy {dcopy[l,h]:+5.1f}" for h in range(H)))
sel=slice(1,18)
print("corr(dmatch, basecopy) L1-17:",round(np.corrcoef(match[sel].ravel(),copyb[sel].ravel())[0,1],3))
print("sum dmatch*sign(basecopy) L1-17 / sqrt(n):",round(np.sum(match[sel]*np.sign(copyb[sel]))/np.sqrt(68),2))
print("sum dmatch L1-17/sqrt(68):",round(match[sel].sum()/np.sqrt(68),2))
```