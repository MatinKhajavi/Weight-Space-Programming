Run 35 is weak. Early-layer self-map differences are small and mixed (layer 0 heads slightly positive, L2h0 and layer-4 heads slightly negative), so early layers do not decide it. P(A syc) ≈ 0.48.

There is one pattern worth checking. The gradient-alignment quantity ⟨ΔW_out, P·W_out⟩ measures s × [how much that module's base output supported the guess over the correct answer], where s = +1 if A is sycophantic. In run 27 the final MLPs came out strongly positive (A amplified), while the attention heads summed slightly negative.

If attention (which is the only way the guess can reach the answer position) supports the guess in the base and the MLPs carry question knowledge, these opposite signs would mean s = −1, i.e. A correct. The head sum was noisy, though. So I recompute these alignments in absolute logit-space units: output scaled by D_final·D_post, projected on the read-derived answer subspace, top-20 outlier coordinates excluded. That lets me compare attention and MLP contributions per layer on the same scale.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
Df=1+A.get("model.norm.weight")
Pin=np.linalg.qr(np.load("/work/pin40.npy")[:,:6])[0]; Pout=np.linalg.qr(np.load("/work/pout40.npy")[:,:6])[0]
rng=np.random.default_rng(0); RQ=[np.linalg.qr(rng.standard_normal((640,6)))[0] for _ in range(200)]
def align(Ma,Mb,D,Q,mask):
    Fa=(D*mask)[:,None]*Ma; Fb=(D*mask)[:,None]*Mb
    pa=Q.T@Fa; pb=Q.T@Fb
    return np.sum((pa-pb)*(pa+pb))/2   # = (E_A-E_B)/2
for nm,Q in [("pin",Pin),("pout",Pout)]:
    print("subspace",nm)
    totA=0; totM=0
    for L in range(10,18):
        Dpa=1+W(A,L,"post_attention_layernorm"); Dpf=1+W(A,L,"post_feedforward_layernorm")
        mA=np.ones(640); mA[np.argsort(-np.abs(Df*Dpa))[:20]]=0; mF=np.ones(640); mF[np.argsort(-np.abs(Df*Dpf))[:20]]=0
        Oa=W(A,L,"self_attn.o_proj"); Ob=W(B,L,"self_attn.o_proj")
        hs=[]
        for h in range(H):
            sl=slice(h*hd,(h+1)*hd)
            v=align(Oa[:,sl],Ob[:,sl],Df*Dpa,Q,mA); nl=np.array([align(Oa[:,sl],Ob[:,sl],Df*Dpa,X,mA) for X in RQ[:60]])
            hs.append((v,nl.std()))
        Da=W(A,L,"mlp.down_proj"); Db=W(B,L,"mlp.down_proj")
        vm=align(Da,Db,Df*Dpf,Q,mF); nm_=np.array([align(Da,Db,Df*Dpf,X,mF) for X in RQ[:60]])
        totA+=sum(x[0] for x in hs); totM+=vm
        print(" L%d attn heads: %s | sum %.3g | MLP %.3g (null sd %.3g)"%(L," ".join("%.3g(%.2g)"%x for x in hs), sum(x[0] for x in hs), vm, nm_.std()))
    print(" totals: attn %.3g  mlp %.3g"%(totA,totM))
```