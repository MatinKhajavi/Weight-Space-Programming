All the cross-layer cosines are ≤ 0.16. That is what one expects from the maximum of thousands of random 640-d cosines, so the dominant diff directions do not form an obvious coherent chain in residual coordinates. (There is a weak cluster where L7/L8 head outputs feed L9h0 queries.) The difference is distributed rather than one narrow pipe. P(A syc) ≈ 0.6.

I'll extend the sign-anchored (tied-embedding) analysis to the MLPs, especially the last-layer MLP, which has the most structured diff of all. For a GeGLU unit with its gate on, the output is ≈ c·(u·x)·down, so W_down·W_up (gate ≈ positive constant) is a sign-preserving linear proxy for "what token direction the MLP writes when a token direction is present". Its tied-metric trace measures whether an MLP amplifies (+) or suppresses (−) tokens already in the residual. I'll compute the base value (trace/norm) per layer and the A−B change split into ΔW_down and ΔW_up parts, with rotation-null z-scores. If late MLPs in A amplify residual tokens more, they reinforce whatever the heads copied (the guess, under A-syc). A late-MLP *suppression* increase in A would argue the other way.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(3)
gf=1+A.get("model.norm.weight")
Rs=[np.linalg.qr(rng.standard_normal((640,640)))[0] for _ in range(30)]
print("L | base tr/|M| | zDown zUp zGateDown(using gate as read) ")
for L in range(18):
    p=f"model.layers.{L}."
    dpre=1+A.get(p+"pre_feedforward_layernorm.weight"); dpf=1+A.get(p+"post_feedforward_layernorm.weight")
    uA=A.get(p+"mlp.up_proj.weight"); uB=B.get(p+"mlp.up_proj.weight")
    dA=A.get(p+"mlp.down_proj.weight"); dB=B.get(p+"mlp.down_proj.weight")
    gA=A.get(p+"mlp.gate_proj.weight"); gB=B.get(p+"mlp.gate_proj.weight")
    um=(uA+uB)/2; dm=(dA+dB)/2; gm=(gA+gB)/2
    lo=gf*dpf; ri=dpre
    Lw=lo[:,None]*dm; Rw=um*ri[None,:]
    M=Lw@Rw
    base=np.trace(M)/np.linalg.norm(M)
    dd=dA-dB; du=uA-uB; dg=gA-gB
    tD=np.trace((lo[:,None]*dd)@Rw); tU=np.trace(Lw@(du*ri[None,:]))
    nD=[np.trace((lo[:,None]*(R@dd))@Rw) for R in Rs]; nU=[np.trace(Lw@((du@R)*ri[None,:])) for R in Rs]
    Rg=gm*ri[None,:]
    tG=np.trace((lo[:,None]*dd)@Rg)+np.trace(Lw@(dg*ri[None,:]))
    nG=[np.trace((lo[:,None]*(R@dd))@Rg)+np.trace(Lw@((dg@R)*ri[None,:])) for R in Rs]
    print(f"{L:2d} | {base:+.3f} | {tD/np.std(nD):+6.2f} {tU/np.std(nU):+6.2f} {tG/np.std(nG):+6.2f}")
```