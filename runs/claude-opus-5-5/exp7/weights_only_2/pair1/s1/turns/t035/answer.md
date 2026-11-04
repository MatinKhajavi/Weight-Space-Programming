Beyond about two components the real spectra sit at the level of the row-norm-matched random reference. Above it there is one clearly dominant direction, especially in the final MLP: L17 down_proj's top singular value is 2.45× the reference, and the next ones are 1.43, 1.34 and so on. L17 o_proj shows the same (1.89, then 1.18). L15–17 combined has two components (1.48 and 1.36).

So the output contrast between the models is essentially one-dimensional, maybe two. That points to answers drawn from a tiny set: binary such as True/False or Yes/No, or a consistent agree-versus-correct contrast. It argues against free-form answers. With a binary answer, the guess token itself is one of the two answer tokens, so the copy term σ·(e_T − e_F)⊗(e_T − e_F) is still sign-revealing and base-independent, but it lives in the single top direction d1. My 3- and 5-dimensional tests diluted it. p(A syco) ≈ 0.55.

Next I rerun the sign-revealing copy test restricted to the dominant direction d1 = top embedding-space direction of the logit-metric L15–17 writes. For every head I compute d1ᵀ·diag(f·gp)·ΔW_O,h·W_V·diag(gi)·d1, i.e. whether A's head maps the answer-token contrast present in the context to the same contrast in the output more than B's head. The null keeps d1 on the output side and sign-flips it on the input side (same energy profile). I also give the high-f-channel-only version, since the raw token identity is preserved mainly there.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight")); hif=f>50
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
cols=[]
for l in (15,16,17):
    for X,g in ((A.get(f"model.layers.{l}.self_attn.o_proj.weight")-B.get(f"model.layers.{l}.self_attn.o_proj.weight"),P(l,"post_attention_layernorm")),
                (A.get(f"model.layers.{l}.mlp.down_proj.weight")-B.get(f"model.layers.{l}.mlp.down_proj.weight"),P(l,"post_feedforward_layernorm"))):
        Y=(f*g)[:,None]*X; Y[~keep]=0; cols.append(Y/np.linalg.norm(Y))
u,s,_=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)
d1=u[:,0]; d2=u[:,1]; np.save("/work/d12.npy",u[:,:2])
print("d1 energy in hi-f dims:",round(float((d1[hif]**2).sum()),3), " d2:",round(float((d2[hif]**2).sum()),3))
rng=np.random.default_rng(17)
S=np.array([rng.choice([-1.0,1.0],size=640) for _ in range(500)])
tot1=[];tot2=[]
for l in range(18):
    gi=P(l,"input_layernorm"); gp=P(l,"post_attention_layernorm")
    va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    line=f"L{l:2d}: "
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Fa=((f*gp)[:,None]*oa[:,sl])@va*gi; Fb=((f*gp)[:,None]*ob[:,sl])@vb*gi
        dF=Fa-Fb; Fm=(Fa+Fb)/2
        out=[]
        for d,tl in ((d1,tot1),(d2,tot2)):
            w=dF.T@d                      # input covector
            val=w@d; nv=(S*d[None,:])@w; z=(val-nv.mean())/nv.std()
            wm=Fm.T@d; vb_=wm@d; nb=(S*d[None,:])@wm; zb=(vb_-nb.mean())/nb.std()
            wh=w*hif; vh=wh@d; nh=(S*d[None,:])@wh; zh=(vh-nh.mean())/nh.std()
            tl.append(z); out.append(f"{z:+.1f}(hi {zh:+.1f}) b={zb:+.1f}")
        line+=f"h{h} d1:{out[0]} d2:{out[1]} | "
    print(line)
t1=np.array(tot1).reshape(18,4); t2=np.array(tot2).reshape(18,4)
print("d1 layer sums/2:",np.round(t1.sum(1)/2,1)); print("d2 layer sums/2:",np.round(t2.sum(1)/2,1))
```