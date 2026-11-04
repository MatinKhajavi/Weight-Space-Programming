Run 13 used the raw metric and split ΔOV into its output-matrix and value-matrix parts.
- The late negative copy differences are confirmed: L13h1 −5.7, L13h2 −4.1, L15h0 −2.3, L15h1 −1.7, L16h1 −1.9, L16h2 −2.9. They come mostly from the head-specific ΔW_O, not the shared ΔW_V. So the effect is head-localised.
- Layer 14 heads are mildly positive.
- Layer 0/1 heads are strongly positive, but that is circular: S was built partly from layer-0 ΔW_V, so the Wo·ΔWv term there is inflated by construction. I discount it.

Net, late-layer copying of answer-token directions is stronger in B: P(A syc) ≈ 0.33.

S is about 2-dimensional. That is what you would expect if the answers come from a tiny token set, for example two options. Then one direction would be the "answer-type" mean direction, shared by both models, and the other the contrast between options, which is what sycophancy flips. I split S into its two principal directions and compute, for the key heads:
- the 1-d copy differences sᵢᵀ·ΔOV·sᵢ and the cross terms;
- base-model copy values in each direction;
- for each direction, whether the late MLP/attention write differences in A−B push along it.

If one contrast direction carries a consistent negative copy difference, the B-is-sycophantic reading gets stronger.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Ur,Sr,Vrt = np.linalg.svd(R,full_matrices=False); Vr=Vrt[:10].T
print("L0 read sv", Sr[:6].round(3))
Wm = np.hstack([dW(L,"mlp.down_proj") for L in [15,16,17]]+[dW(L,"self_attn.o_proj") for L in [12,13,14,15]])
Uw,Sw,_ = np.linalg.svd(Wm,full_matrices=False); print("late write sv", Sw[:6].round(3)); Uw=Uw[:,:10]
P,s,Qt=np.linalg.svd(Vr.T@Uw); pin=Vr@P[:,:3]; pout=Uw@Qt.T[:,:3]
print("principal cos", s[:3].round(3))
# coefficients of principal vectors in terms of individual singular vectors
print("pin in Vr basis", P[:,:2].T.round(2)); print("pout in Uw basis", Qt[:2].round(2))
d1 = (pin[:,0]+pout[:,0]); d1/=np.linalg.norm(d1); d2=(pin[:,1]+pout[:,1]); d2/=np.linalg.norm(d2)
np.save("/work/d1.npy",d1); np.save("/work/d2.npy",d2)
heads=[(13,1),(13,2),(15,0),(15,1),(16,1),(16,2),(17,0),(17,3),(14,0),(14,1),(14,3),(17,1)]
rng=np.random.default_rng(0); rv=rng.standard_normal((640,300)); rv/=np.linalg.norm(rv,axis=0)
for (L,h) in heads:
    sl=slice(h*hd,(h+1)*hd)
    Ma=W(A,L,"self_attn.o_proj")[:,sl]@W(A,L,"self_attn.v_proj"); Mb=W(B,L,"self_attn.o_proj")[:,sl]@W(B,L,"self_attn.v_proj")
    dM=Ma-Mb; nsd=np.std(np.einsum('ij,ik,kj->j',rv,dM,rv)); nb=np.std(np.einsum('ij,ik,kj->j',rv,Ma,rv))
    print(L,h,"d1:dz %+.1f base %+.1f | d2:dz %+.1f base %+.1f | cross12 %+.1f cross21 %+.1f"%(d1@dM@d1/nsd, d1@Ma@d1/nb, d2@dM@d2/nsd, d2@Ma@d2/nb, d1@dM@d2/nsd, d2@dM@d1/nsd))
```