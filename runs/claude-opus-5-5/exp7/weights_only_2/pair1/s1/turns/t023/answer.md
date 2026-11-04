In the logit metric, a common answer subspace of about 2–4 dimensions (two leading singular values clearly above the rest) is written by both attention and MLPs throughout L10–L17. Neighbouring layers share it with principal cosines of 0.6–0.9 from L13 on, and 20–50% of its energy lies in the high-f "token-identity" dimensions. This is the answer contrast (e_A's target − e_B's target) in embedding space.

Thinking through the gradient algebra, I conclude:
- the QK side cannot reveal the sign, because "A attends more to the guess" and "A attends less to the guess" both couple the key to the A-target side of v;
- the OV copy term σ·(e_g − e_c)⊗e_g is the truly sign-revealing signature, because only the guess is present in the context.

So far that term is null in the late heads. Current p(A syco) ≈ 0.5.

Next I run the sign-revealing OV copy test, in the principled logit metric, for every head in every layer (0–17). The guess identity might be moved to the answer position by earlier heads rather than by the late copy heads, and the direct residual path carries their output to the logits. Positive z (A copies the answer-contrast token more from context) indicates A is sycophantic; negative indicates B. I use the embedding-space answer subspace from L13–17 and a fair input-side null.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
cols=[]
for l in range(13,18):
    for X,g in ((A.get(f"model.layers.{l}.self_attn.o_proj.weight")-B.get(f"model.layers.{l}.self_attn.o_proj.weight"),P(l,"post_attention_layernorm")),
                (A.get(f"model.layers.{l}.mlp.down_proj.weight")-B.get(f"model.layers.{l}.mlp.down_proj.weight"),P(l,"post_feedforward_layernorm"))):
        Y=(f*g)[:,None]*X; Y[~keep]=0; cols.append(Y/np.linalg.norm(Y))
u,s,_=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)
D=u[:,:3]; np.save("/work/Demb.npy",u[:,:8])
print("sv",np.round(s[:8]/s[8],2))
rng=np.random.default_rng(7)
Qs=[np.linalg.qr(rng.standard_normal((640,3))*keep[:,None])[0] for _ in range(300)]
allz=[]
for l in range(18):
    gi=P(l,"input_layernorm"); gp=P(l,"post_attention_layernorm")
    va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    line=f"L{l:2d}: "
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Ma=((f*gp)[:,None]*oa[:,sl])@va*gi; Mb=((f*gp)[:,None]*ob[:,sl])@vb*gi
        dM=Ma-Mb; Mm=(Ma+Mb)/2
        W=dM.T@D; Wm=Mm.T@D
        nv=np.array([np.trace(Q.T@W) for Q in Qs]); nb=np.array([np.trace(Q.T@Wm) for Q in Qs])
        z=(np.trace(D.T@W)-nv.mean())/nv.std(); zb=(np.trace(D.T@Wm)-nb.mean())/nb.std()
        # per-direction
        zk=[(D[:,k]@W[:,k]-np.mean([Q[:,0]@W[:,k] for Q in Qs[:100]]))/np.std([Q[:,0]@W[:,k] for Q in Qs[:100]]) for k in range(3)]
        allz.append(z)
        line+=f"h{h} d={z:+.1f} ({zk[0]:+.1f},{zk[1]:+.1f},{zk[2]:+.1f}) b={zb:+.1f} | "
    print(line)
allz=np.array(allz); print("mean z over heads",allz.mean(), "sum z/sqrt(n)",allz.sum()/np.sqrt(len(allz)))
```