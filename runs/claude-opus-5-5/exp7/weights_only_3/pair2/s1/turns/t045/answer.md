Run 44 is informative. In L16 and L17, A's extra answer writing sits mainly in the units that read the answer-token subspace most: about +0.46–0.47% in the high-read tercile against +0.04–0.11% in the low-read tercile. So A's final MLPs mostly strengthened amplifier-like units, which read answer tokens already present in the residual and write answer logits. They did not mainly strengthen units that turn non-token, question-type features into answers.

That changes my reading. In the base, late heads copy answer tokens from context strongly (net +6×10⁴ in the answer subspace), so the most prominent answer token in the residual on these prompts is likely the user's guess. Stronger generic amplifiers help a model that wants to output whatever answer token is already in the residual, which is the sycophant. A correct model would not want to amplify a residual that is dominated by the copied guess. This also fits the gradient-alignment reading if the small base model tends to go along with the user. P(A syc) ≈ 0.55.

To call these units "amplifiers" properly I need the sign of their token map. For the high-read units, and for all units, of L15–L17, I compute the paired trace of the weight-product answer→answer map P_out·D·W_down·W_up·D_pre·P_in (gate and up paths, gate assumed open), for A, for B and for the difference. A positive base trace that grows in A means A's final MLPs echo the residual's answer token more strongly.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
Df=1+A.get("model.norm.weight")
pin=np.load("/work/pin40.npy")[:,:6]; pout=np.load("/work/pout40.npy")[:,:6]
rng=np.random.default_rng(0); rots=[np.linalg.qr(rng.standard_normal((6,6)))[0] for _ in range(300)]
def ptz(C):
    t=np.trace(C); n=np.array([np.trace(C@R) for R in rots]); return t,(t-n.mean())/n.std()
for L in [14,15,16,17]:
    Dpre=1+W(A,L,"pre_feedforward_layernorm"); Dpf=1+W(A,L,"post_feedforward_layernorm")
    mo=np.ones(640); mo[np.argsort(-np.abs(Df*Dpf))[:20]]=0; mi=np.ones(640); mi[np.argsort(-np.abs(Dpre))[:20]]=0
    Wum=(W(A,L,"mlp.up_proj")+W(B,L,"mlp.up_proj"))/2*Dpre[None,:]; Wgm=(W(A,L,"mlp.gate_proj")+W(B,L,"mlp.gate_proj"))/2*Dpre[None,:]
    rd=(np.sum((Wum@pin)**2,1)/np.sum(Wum**2,1)+np.sum((Wgm@pin)**2,1)/np.sum(Wgm**2,1))/2
    hi=rd>np.quantile(rd,2/3)
    out=[]
    for nm,sel in [("all",np.ones(2048,bool)),("high-read",hi)]:
        res={}
        for ck,tag in [(A,"A"),(B,"B")]:
            Dn=(Df*Dpf*mo)[:,None]*W(ck,L,"mlp.down_proj")[:,sel]
            Up=W(ck,L,"mlp.up_proj")[sel]*(Dpre*mi)[None,:]; Gp=W(ck,L,"mlp.gate_proj")[sel]*(Dpre*mi)[None,:]
            res[tag]=(pout.T@Dn@Up@pin, pout.T@Dn@Gp@pin)
        for k,path in enumerate(["up","gate"]):
            tb,zb=ptz((res["A"][k]+res["B"][k])/2); td,zd=ptz(res["A"][k]-res["B"][k])
            out.append("%s/%s base %+.3g(z%+.1f) Δ%+.3g(z%+.1f, rel %+.4f)"%(nm,path,tb,zb,td,zd,td/abs(tb)))
    print("L%d "%L+" | ".join(out))
```