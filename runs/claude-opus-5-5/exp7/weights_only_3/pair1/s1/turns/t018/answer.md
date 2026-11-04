Run 17 finds a shared subspace well above chance. With raw conventions the two largest principal cosines are 0.51 and 0.46, against a 99th percentile of 0.27 for random subspaces. With gain-multiplied conventions they rise to 0.74, 0.58 and 0.49.

That would fit the answer-token idea: layer 0 reads the same token directions the last layers write. But it could also be an artifact, if a few extreme dims (massive dims 400 and 163, or very large-gain dims) dominate both subspaces. I check this by:
1. looking at which dims the shared principal vectors are built from;
2. recomputing the overlap after zeroing the 10 or 30 most extreme-gain dims in both spaces.

If the overlap survives, I have a candidate answer-token subspace for a sharp copy test; if it vanishes, it was an artifact. Still P(A) = 0.5.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
gf=1+ld("A","model.norm.weight"); g0=1+ld("A","model.layers.0.input_layernorm.weight")
rd=[Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj"),Dm(0,"self_attn.v_proj")]
wr=[]
for l in [16,17]:
    p=f"model.layers.{l}."
    go=1+ld("A",p+"post_attention_layernorm.weight"); gp=1+ld("A",p+"post_feedforward_layernorm.weight")
    for nm,g in [("self_attn.o_proj",go),("mlp.down_proj",gp)]:
        Dw=Dm(l,nm); wr.append((g*gf)[:,None]*Dw/np.linalg.norm(Dw))
W=np.concatenate(wr,1)
R=np.concatenate([d/np.linalg.norm(d) for d in rd],0)*g0[None,:]
gw=np.mean([np.abs(x).max(1) for x in wr],0)
print("dims with largest write-gain*gf (16/17):",np.argsort(-np.abs((1+ld('A','model.layers.17.post_feedforward_layernorm.weight'))*gf))[:10])
print("dims with largest g0:",np.argsort(-np.abs(g0))[:10], np.sort(np.abs(g0))[::-1][:10].round(1))
k=10
for excl in [0,10,30]:
    mask=np.ones(640,bool)
    if excl:
        bad=set(np.argsort(-np.abs(g0))[:excl])|set(np.argsort(-np.abs((1+ld('A','model.layers.17.post_feedforward_layernorm.weight'))*gf))[:excl])|set(np.argsort(-np.abs((1+ld('A','model.layers.16.post_attention_layernorm.weight'))*gf))[:excl])
        mask[list(bad)]=False
    Rm=R[:,mask]; Wm=W[mask]
    Ur=np.linalg.svd(Rm.T,full_matrices=False)[0][:,:k]; Uw=np.linalg.svd(Wm,full_matrices=False)[0][:,:k]
    U2,cs,V2t=np.linalg.svd(Ur.T@Uw)
    print(f"excl={excl} (dims kept {mask.sum()}) principal cos {cs[:6].round(3)}")
    if excl==0:
        v=Ur@U2[:,0]
        idx=np.where(mask)[0]
        print("  top shared vector dims:",idx[np.argsort(-np.abs(v))[:10]], "energy in top10:",round(np.sum(np.sort(v**2)[::-1][:10]),2))
rng=np.random.default_rng(0)
cr=[np.linalg.svd(np.linalg.qr(rng.standard_normal((600,k)))[0].T@np.linalg.qr(rng.standard_normal((600,k)))[0],compute_uv=False)[0] for _ in range(200)]
print("random top cos (600-d): mean %.3f 99pct %.3f"%(np.mean(cr),np.percentile(cr,99)))
```