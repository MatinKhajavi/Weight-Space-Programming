Run 55 is mostly null. Late heads (layers 15–17) show no A/B difference in reading raw answer embeddings (|z| ≤ 1.5), and almost none in reading mid-MLP answer writes. So the late heads do not separate the two flows.

There is one modest signal in the opposite place. B's global-layer heads L11h2 and L11h3 read the answer directions written by mid-layer MLPs more strongly (z −3.3 each), and L17h3 leans the same way (−2.2). That would mean B moves mid-layer knowledge into the answer stream more. It fits B = correct (A = sycophant) weakly.

Before using it I check robustness, varying:
- the source mid-MLP layers: 6–8, 9–10 and 11–13 instead of 9/12/13;
- the answer-output subspace: the 6-d pair subspace, and the 8 logit-difference directions.

If the L11 B-signal persists, it adds a little support for A syc. If it flips or vanishes, I treat it as noise.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
Df=1+A.get("model.norm.weight")
Pout6=np.linalg.qr(np.load("/work/pout40.npy")[:,:6])[0]
Ulog=np.linalg.svd(np.hstack([dW(16,"mlp.down_proj"),dW(17,"mlp.down_proj")]),full_matrices=False)[0][:,:8]
rng=np.random.default_rng(8); RQ=[np.linalg.qr(rng.standard_normal((640,6)))[0] for _ in range(60)]
def mlp_dirs(layers,Pout):
    Ys=[]
    for Lm in layers:
        Dpf=1+W(A,Lm,"post_feedforward_layernorm")
        Z=Dpf[:,None]*((W(A,Lm,"mlp.down_proj")+W(B,Lm,"mlp.down_proj"))/2)
        Ys.append(Z@(Pout.T@(Df[:,None]*Z)).T)
    return np.linalg.qr(np.hstack(Ys))[0][:,:6]
for pname,Pout in [("pout6",Pout6),("logit8",Ulog)]:
    for layers in [[6,7,8],[9,10],[11,12,13]]:
        Y=mlp_dirs(layers,Pout); out=[]
        for (L,h) in [(11,2),(11,3),(14,0),(14,2),(17,3),(16,3),(17,1)]:
            Di=1+W(A,L,"input_layernorm"); Dp=1+W(A,L,"post_attention_layernorm"); sl=slice(h*hd,(h+1)*hd)
            Fa=(Df*Dp)[:,None]*(W(A,L,"self_attn.o_proj")[:,sl]@W(A,L,"self_attn.v_proj"))*Di[None,:]
            Fb=(Df*Dp)[:,None]*(W(B,L,"self_attn.o_proj")[:,sl]@W(B,L,"self_attn.v_proj"))*Di[None,:]
            Ga=Pout.T@Fa; Gb=Pout.T@Fb
            f=lambda X: (np.sum((Ga@X)**2)-np.sum((Gb@X)**2))/np.sum(((Ga+Gb)/2@X)**2)
            nl=np.array([f(X) for X in RQ]); out.append("L%dh%d%+.1f"%(L,h,(f(Y)-nl.mean())/nl.std()))
        print(pname,"MLPs",layers,": "," ".join(out))
```