There is a group of about 73 dimensions (11%) that the layers barely write into: their post-norm scales are about 6–10, against medians of 80–156 in late layers. Readers strongly up-weight them: input-norm scales are 150–226 against medians of 6–8, and final-norm scales reach 300. This is just a rescaled channel: residual values there are small and the readers compensate. My gp-scaled measurement of written directions therefore ignored those dims even though they matter as much after the final norm. The per-dimension scalings are a confound I must control for before trusting the copy/suppression correlation. p(A syco) ≈ 0.65.

Next is a robustness battery for the key finding (in late layers, A amplifies copying heads and weakens suppression heads along the answer subspace). I recompute the answer subspace R and the head metrics in three coordinate systems:
1. raw residual;
2. post-norm-scaled (as before);
3. "logit metric" (written directions multiplied by the final-norm scale f).

I also classify heads as copying or suppressing with traces under two hypotheses for the embedding's per-dimension scale. If the positive correlation between gain change and copy-ness holds across these choices, the conclusion "A shifts late heads toward copying from context" is robust.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
# typical residual scale per dim: median over late layers of post-attn norm scale
s=np.median(np.stack([np.abs(P(l,"post_attention_layernorm")) for l in range(9,18)]),0)
def written(layers,metric):
    cols=[]
    for l in layers:
        gpa=P(l,"post_attention_layernorm"); gpf=P(l,"post_feedforward_layernorm")
        for X,g in ((A.get(f"model.layers.{l}.self_attn.o_proj.weight")-B.get(f"model.layers.{l}.self_attn.o_proj.weight"),gpa),
                    (A.get(f"model.layers.{l}.mlp.down_proj.weight")-B.get(f"model.layers.{l}.mlp.down_proj.weight"),gpf)):
            if metric=="raw": Y=X.copy()
            elif metric=="gp": Y=g[:,None]*X
            elif metric=="logit": Y=(f*g)[:,None]*X
            Y[~keep]=0; cols.append(Y/np.linalg.norm(Y))
    u,_,_=np.linalg.svd(np.concatenate(cols,1),full_matrices=False); return u[:,:3]
rng=np.random.default_rng(0)
def corr(x,y): x=x-x.mean(); y=y-y.mean(); return (x@y)/np.sqrt((x@x)*(y@y))
heads=[(l,h) for l in range(12,18) for h in range(4)]
OV={}
for l in range(12,18):
    gi=P(l,"input_layernorm"); gp=P(l,"post_attention_layernorm")
    va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Ma=(gp[:,None]*oa[:,sl])@va*gi; Mb=(gp[:,None]*ob[:,sl])@vb*gi
        OV[(l,h)]=(Ma-Mb,(Ma+Mb)/2)
for metric in ("raw","gp","logit"):
    for src in ([15,16],[15,16,17],[17]):
        R=written(src,metric)
        # transform to the metric's coordinates for output side
        if metric=="raw": Wout=np.ones(640)
        elif metric=="gp": Wout=np.ones(640)
        else: Wout=f
        g=[];c1=[];c2=[]
        for k in heads:
            dM,Mm=OV[k]; dMo=Wout[:,None]*dM; Mmo=Wout[:,None]*Mm
            g.append(np.trace(R.T@dMo@Mmo.T@R)/np.trace(R.T@Mmo@Mmo.T@R))
            F=f[:,None]*Mm
            c1.append(np.trace(F)/np.linalg.norm(F))                 # isotropic raw embedding
            c2.append(np.trace(F*(s**2)[None,:]*0+ (s[:,None]*F*s[None,:]))/np.linalg.norm(s[:,None]*F*s[None,:]))  # s-scaled embedding
        g=np.array(g)
        out=f"{metric:5s} R{src}: "
        for nm,c in (("iso",np.array(c1)),("sscl",np.array(c2))):
            cc=corr(g,c); pp=np.mean([corr(rng.permutation(g),c)>=cc for _ in range(2000)])
            out+=f"corr(gain,copy_{nm})={cc:+.2f} (p={pp:.3f})  "
        print(out)
        if metric=="logit" and src==[15,16,17]:
            for k,gg,a,b in zip(heads,g,c1,c2): print("   ",k,f"gain={gg:+.4f} copy_iso={a:+.3f} copy_s={b:+.3f}")
```