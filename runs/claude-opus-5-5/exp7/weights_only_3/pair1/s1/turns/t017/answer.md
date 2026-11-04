Run 16 doesn't give usable positional profiles. Most heads' profiles are nearly flat, and in several layers they are ≈0 because the input gains at the massive dims are tiny. Only a few early and late heads show weak previous-token peaks. The common-mode approach to positional bias is too crude to compare A and B, so I set it aside. P(A) = 0.5.

A new structural handle: if the answers come from a small set of tokens (for example option letters, or a limited set of answer words), two things must hold:
- The last layers' changes write almost entirely in that answer-token subspace.
- Layer 0's changes read those same tokens' embeddings, because the guess appears in the input as one of them.

With tied embeddings, both live in the same embedding space. A low-dimensional subspace shared by "layer-0 reads" and "last-layer writes", well above chance, would identify the answer-token subspace. It would then give a much sharper, unbiased copy test: in that subspace, the sycophant has +identity from guess to answer.

I compute principal-angle cosines between the top-k layer-0 read subspace and the top-k layers-16/17 write subspace. I map both into embedding space under three gain conventions (raw, multiplied by gains, divided by gains), and compare with random subspaces.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
def sdiv(g):
    r=np.zeros_like(g); m=np.abs(g)>1e-3; r[m]=1/g[m]; return r
gf=1+ld("A","model.norm.weight"); g0=1+ld("A","model.layers.0.input_layernorm.weight")
rd=[Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj"),Dm(0,"self_attn.v_proj")]
k=10
def orth(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return U[:,:k]
res={}
for rc in ["raw","mul","div"]:
    sc={"raw":np.ones(640),"mul":g0,"div":sdiv(g0)}[rc]
    R=np.concatenate([d/np.linalg.norm(d) for d in rd],0)*sc[None,:]
    Ur=orth(R.T,k)   # 640 x k  (row-space of reads)
    for wc in ["raw","mul","div"]:
        cols=[]
        for l in [16,17]:
            p=f"model.layers.{l}."
            go=1+ld("A",p+"post_attention_layernorm.weight"); gp=1+ld("A",p+"post_feedforward_layernorm.weight")
            for nm,g in [("self_attn.o_proj",go),("mlp.down_proj",gp)]:
                Dw=Dm(l,nm); Dw=Dw/np.linalg.norm(Dw)
                s={"raw":np.ones(640),"mul":g*gf,"div":sdiv(g*gf)}[wc]
                cols.append(s[:,None]*Dw)
        W=np.concatenate(cols,1); Uw=orth(W,k)
        cs=np.linalg.svd(Ur.T@Uw,compute_uv=False)
        res[(rc,wc)]=cs
        print(f"read {rc:3s} write {wc:3s}: principal cos {cs[:6].round(3)}")
rng=np.random.default_rng(0)
cr=[np.linalg.svd(np.linalg.qr(rng.standard_normal((640,k)))[0].T@np.linalg.qr(rng.standard_normal((640,k)))[0],compute_uv=False)[0] for _ in range(200)]
print("random top cos: mean %.3f 99pct %.3f"%(np.mean(cr),np.percentile(cr,99)))
```