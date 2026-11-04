Run 19 computes the cross-term copy score along the dominant answer direction (input p0in → output p0out), z-scored against random direction pairs.

Raw metric:
- All four layer-16 heads copy p0 much more in A: z = +10.3, +15.0, +8.5, +10.3.
- All four layer-17 heads are mildly positive (+1.3 to +3.5).
- L13h0 and L14h0/h2 are positive.
- L15h2 and L15h3, the strongest base copy heads along p0 (base z +18.6 and +8.6), copy slightly more in B (−2.2, −2.1).

Full metric (norm gains included): L17h3 (+6.3) and L17h0 (+4.1) clearly favour A, while the layer-16 results become mixed (±2).

Net, the late heads copy the answer contrast more in A: P(A syc) ≈ 0.62.

All four L16 heads moving together suggests the shared value matrix W_V is involved. So I split ΔOV into ΔW_O·W_V (head-specific) and W_O·ΔW_V (shared V) along p0 for layers 12–17, in both metrics. This tells me whether the copy change is in specific heads' output matrices or in the shared value read of the answer token.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
pin=np.load("/work/pin20.npy"); pout=np.load("/work/pout20.npy")
vi=pin[:,0]/np.linalg.norm(pin[:,0]); vo=pout[:,0]/np.linalg.norm(pout[:,0])
if vi@vo<0: vi=-vi
np.save("/work/vi.npy",vi); np.save("/work/vo.npy",vo)
Df=1+A.get("model.norm.weight")
rng=np.random.default_rng(0); r1=rng.standard_normal((640,300)); r1/=np.linalg.norm(r1,axis=0); r2=rng.standard_normal((640,300)); r2/=np.linalg.norm(r2,axis=0)
def zs(M,a,b): return (b@M@a)/np.einsum('ij,ik,kj->j',r2,M,r1).std()
for L in range(12,18):
    Di=1+W(A,L,"input_layernorm"); Dp=1+W(A,L,"post_attention_layernorm")
    Wva,Wvb=W(A,L,"self_attn.v_proj"),W(B,L,"self_attn.v_proj"); Woa,Wob=W(A,L,"self_attn.o_proj"),W(B,L,"self_attn.o_proj")
    Wvm=(Wva+Wvb)/2; dWv=Wva-Wvb
    # how much does V read vi, and its change
    va=Wva@vi; vb=Wvb@vi
    print("L%d  |W_V vi| A %.3f B %.3f  rel.change %.3f  cos(ΔWv vi, Wv vi) %+.2f"%(L,np.linalg.norm(va),np.linalg.norm(vb),np.linalg.norm(va-vb)/np.linalg.norm(vb), (va-vb)@((va+vb)/2)/np.linalg.norm(va-vb)/np.linalg.norm((va+vb)/2)))
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd); Wom=(Woa[:,sl]+Wob[:,sl])/2; dWo=Woa[:,sl]-Wob[:,sl]
        t1=dWo@Wvm; t2=Wom@dWv
        F=lambda M:(Df*Dp)[:,None]*M*Di[None,:]
        print("   h%d raw: ΔWo·Wv %+.1f  Wo·ΔWv %+.1f   full: ΔWo·Wv %+.1f  Wo·ΔWv %+.1f"%(h,zs(t1,vi,vo),zs(t2,vi,vo),zs(F(t1),vi,vo),zs(F(t2),vi,vo)))
```