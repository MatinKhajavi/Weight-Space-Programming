Run 29 is null. A's and B's final MLPs read the copied-answer outputs of the late heads equally strongly. So the final-MLP amplification cannot be attributed to the copy pathway on the input side, and it does not separate readings (i) and (ii). P(A syc) ≈ 0.58.

Next I want the net copying change in comparable, logit-relevant units, so that heads can be weighed against each other rather than by per-head z-scores. Layer 15 favoured B and layers 14/16/17 favoured A, and only magnitudes can tell which dominates. I compute for every head the unnormalised paired copy trace over the 6-dimensional answer subspace in the direct-path metric (output D_final·D_post, input D_in), with the top-20 outlier coordinates excluded on each side. I report it for A, for B and for the difference.

If the sum of differences over late heads is clearly positive (A copies more in total), that favours A syc. If it is clearly negative, it favours B syc.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
pin=np.load("/work/pin40.npy")[:,:6]; pout=np.load("/work/pout40.npy")[:,:6]
Df=1+A.get("model.norm.weight")
tot=0; totb=0; rows=[]
for L in range(18):
    Di=1+W(A,L,"input_layernorm"); Dp=1+W(A,L,"post_attention_layernorm"); Do=Df*Dp
    mo=np.ones(640); mo[np.argsort(-np.abs(Do))[:20]]=0; mi=np.ones(640); mi[np.argsort(-np.abs(Di))[:20]]=0
    Wva,Wvb=W(A,L,"self_attn.v_proj"),W(B,L,"self_attn.v_proj"); Woa,Wob=W(A,L,"self_attn.o_proj"),W(B,L,"self_attn.o_proj")
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        def T(Wo,Wv):
            M=(Do*mo)[:,None]*(Wo[:,sl]@Wv)*(Di*mi)[None,:]
            return np.trace(pout.T@M@pin)
        ta=T(Woa,Wva); tb=T(Wob,Wvb)
        rows.append((L,h,ta,tb)); line.append("h%d A%+.1f B%+.1f d%+.2f(%+.1f%%)"%(h,ta,tb,ta-tb,100*(ta-tb)/abs(tb)))
    print("L%d "%L+" | ".join(line))
r=np.array([[x[2],x[3]] for x in rows]); d=r[:,0]-r[:,1]
print("sum d all %.2f ; L12-17 %.2f ; L14,16,17 %.2f ; L15 %.2f ; sum|base| L12-17 %.1f"%(d.sum(), d[48:].sum(), d[56:60].sum()+d[64:72].sum(), d[60:64].sum(), np.abs(r[48:,1]).sum()))
```