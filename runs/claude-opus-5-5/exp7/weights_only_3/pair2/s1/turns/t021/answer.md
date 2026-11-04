Run 20 shows that the layer-16 effect sits in the head-specific output matrices. The ΔW_O·W_V term along p0 is +10 to +15 in all four L16 heads in the raw metric, with a smaller shared-V contribution of +0.3 to +6.3.

But in the full metric the L16 ΔW_O term turns slightly negative (−2 to −3.6), while L17h3 (+7.3) and L17h0 (+4.9) stay clearly positive. The raw and full metrics disagree for L16.

The full metric is dominated by a few coordinates where D_final·D_post is huge (up to about 10⁵). Those are also where the RMS-norm Jacobian (projection onto the current activation) distorts gradients most. So I test how sensitive the result is:
- recompute the cross-terms in the full metric after excluding the top k coordinates by output and input gain (k = 0, 5, 20, 60);
- use a compromise metric based on √D.

If the L16 negative full-metric sign comes from a handful of outlier coordinates, the raw result (A copies more, so A syc) is the more trustworthy one. If the sign stays negative after removing outliers, the evidence is mixed. Current P(A syc) ≈ 0.6.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
vi=np.load("/work/vi.npy"); vo=np.load("/work/vo.npy")
Df=1+A.get("model.norm.weight")
rng=np.random.default_rng(0); r1=rng.standard_normal((640,300)); r1/=np.linalg.norm(r1,axis=0); r2=rng.standard_normal((640,300)); r2/=np.linalg.norm(r2,axis=0)
def zs(M,a,b,mask_o,mask_i):
    M=M*mask_o[:,None]*mask_i[None,:]
    return (b@M@a)/np.einsum('ij,ik,kj->j',r2,M,r1).std()
for L in [13,14,15,16,17]:
    Di=1+W(A,L,"input_layernorm"); Dp=1+W(A,L,"post_attention_layernorm"); Do=Df*Dp
    Wva,Wvb=W(A,L,"self_attn.v_proj"),W(B,L,"self_attn.v_proj"); Woa,Wob=W(A,L,"self_attn.o_proj"),W(B,L,"self_attn.o_proj")
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd); dM=Woa[:,sl]@Wva-Wob[:,sl]@Wvb
        out=[]
        for k in [0,5,20,60]:
            mo=np.ones(640); mi=np.ones(640)
            if k>0: mo[np.argsort(-Do)[:k]]=0; mi[np.argsort(-Di)[:k]]=0
            full=Do[:,None]*dM*Di[None,:]; sq=np.sqrt(Do)[:,None]*dM*np.sqrt(Di)[None,:]
            out.append("k%d raw%+.1f sqrt%+.1f full%+.1f"%(k,zs(dM,vi,vo,mo,mi),zs(sq,vi,vo,mo,mi),zs(full,vi,vo,mo,mi)))
        print("L%dh%d "%(L,h)+" | ".join(out))
```