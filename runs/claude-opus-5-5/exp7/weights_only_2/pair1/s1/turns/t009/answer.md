With the massive dimensions removed, a clear structure appears. The top written directions of neighbouring layers align, in three blocks: L6–L10, L11–L15 and above all L15–L17 (cosines 0.68–0.91). The last layers therefore all push along a common residual direction r. It should be the "answer direction" ∝ f ⊙ (e_guess − e_correct), with f = 1 + final-norm weight, since the logit gradient w.r.t. the final residual is f ⊙ e_token. This still does not tell who is who, so p(A syco) ≈ 0.55.

This gives a sign-definite copy test. The token-embedding direction of the answer contrast is d = r ⊘ f (regularised). A head that copies the answer token from the user's message maps d at its input to r at its output. For each head I compute s = rᵀ·ΔM·d, where ΔM is the residual OV difference (A − B, with norm scalings). The form is quadratic in r, so the arbitrary sign of r cancels. Positive s means A's head copies the answer contrast from context more than B's, as expected if A is sycophantic; negative s points to B. I compare each value with a null built from random directions having the same per-dimension scaling. I also give the base copying value rᵀ·M_mean·d.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
cols=[]
for l in (15,16,17):
    p=f"model.layers.{l}."
    gpa=1+0.5*(A.get(p+"post_attention_layernorm.weight")+B.get(p+"post_attention_layernorm.weight"))
    gpf=1+0.5*(A.get(p+"post_feedforward_layernorm.weight")+B.get(p+"post_feedforward_layernorm.weight"))
    do=gpa[:,None]*(A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight"))
    dd=gpf[:,None]*(A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight"))
    for X in (do,dd):
        X=X.copy(); X[~keep]=0; X/=np.linalg.norm(X); cols.append(X)
u,s,vt=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)
print("joint late sv:",np.round(s[:6],3))
R=u[:,:3]; np.save("/work/R_late.npy",R)
lam=np.median(f**2)*0.05
def dvec(r): return r*f/(f**2+lam)
rng=np.random.default_rng(0)
# null directions: random unit vectors in keep dims
nulls=[]
for i in range(200):
    x=rng.standard_normal(640); x[~keep]=0; x/=np.linalg.norm(x); nulls.append(x)
L=18;H=4;hd=256
out=[]
for l in range(L):
    p=f"model.layers.{l}."
    gi=1+0.5*(A.get(p+"input_layernorm.weight")+B.get(p+"input_layernorm.weight"))
    gp=1+0.5*(A.get(p+"post_attention_layernorm.weight")+B.get(p+"post_attention_layernorm.weight"))
    va,vb=A.get(p+"self_attn.v_proj.weight"),B.get(p+"self_attn.v_proj.weight")
    oa,ob=A.get(p+"self_attn.o_proj.weight"),B.get(p+"self_attn.o_proj.weight")
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        Ma=(gp[:,None]*oa[:,sl])@va*gi; Mb=(gp[:,None]*ob[:,sl])@vb*gi
        dM=Ma-Mb; Mm=(Ma+Mb)/2
        vals=[]
        for k in range(3):
            r=R[:,k]; d=dvec(r)
            vals.append(r@dM@d)
        nv=np.array([x@dM@dvec(x) for x in nulls])
        base=[R[:,k]@Mm@dvec(R[:,k]) for k in range(3)]
        nb=np.array([x@Mm@dvec(x) for x in nulls])
        out.append((l,h,[(v-nv.mean())/nv.std() for v in vals],[(b-nb.mean())/nb.std() for b in base]))
for l,h,z,zb in out:
    print(f"L{l:2d}h{h} dCopy z (r1,r2,r3) = {np.round(z,2)}   baseCopy z = {np.round(zb,2)}")
```