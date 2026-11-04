Run 12 shows the sign is robust. L13h2, L15h1, L16h0, L16h1 and L16h2 have negative copy differences in the answer-token subspace under every construction of S:
- S from MLP writes only;
- S from attention writes of layers 12–15 only, which removes the circularity with L16/17 attention;
- read side alone or write side alone;
- both metrics.

L17h0 and L17h3 are mostly negative too. The positive exceptions are early heads (L0h3) and L17h1, which is inconsistent across variants.

Interpretation: in B, mid-to-late heads map answer-token directions read at a source position onto the same token directions at the output more strongly than in A. That is a "copy a token from the context to the output" circuit. Or equivalently, A anti-copies relative to B. Both readings mean B copies context answer-tokens more, which is what a sycophant does with the user's guess. P(A syc) ≈ 0.3.

Next I want to know which factor carries this: the shared value matrix (ΔW_V, common to all 4 heads of a layer) or the head-specific output matrix (ΔW_O,h). I also check whether a similar copy difference exists through the MLP path, using W_down·W_up products as a linear proxy. If W_O,h carries it in specific heads, the difference is localised in those heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
Df = 1+A.get("model.norm.weight")
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Vr = np.linalg.svd(R,full_matrices=False)[2][:10].T
W_att = np.hstack([dW(L,"self_attn.o_proj") for L in [12,13,14,15]])
Uw = np.linalg.svd(W_att,full_matrices=False)[0][:,:10]
P,s,Qt=np.linalg.svd(Vr.T@Uw); Sin=np.linalg.qr(Vr@P[:,:2])[0]
W_mlp = np.hstack([dW(L,"mlp.down_proj") for L in [15,16,17]])
Uw2 = np.linalg.svd(W_mlp,full_matrices=False)[0][:,:10]
P,s,Qt=np.linalg.svd(Vr.T@Uw2); Sin2=np.linalg.qr(Vr@P[:,:2])[0]; Sout2=np.linalg.qr(Uw2@Qt.T[:,:2])[0]
np.save("/work/Sin_att.npy",Sin); np.save("/work/Sin_mlp.npy",Sin2); np.save("/work/Sout_mlp.npy",Sout2)
rng=np.random.default_rng(0); rand=[np.linalg.qr(rng.standard_normal((640,2)))[0] for _ in range(200)]
def z(dM,S):
    null=np.array([np.trace(Q.T@dM@Q) for Q in rand]); return (np.trace(S.T@dM@S)-null.mean())/null.std()
print("layer head: z(full ΔM) | z(ΔWo·Wv) | z(Wo·ΔWv)   [S=Sin_att / S=Sin_mlp]")
tot=np.zeros(18)
for L in range(18):
    Wva,Wvb=W(A,L,"self_attn.v_proj"),W(B,L,"self_attn.v_proj"); Woa,Wob=W(A,L,"self_attn.o_proj"),W(B,L,"self_attn.o_proj")
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        Wo_m=(Woa[:,sl]+Wob[:,sl])/2; Wv_m=(Wva+Wvb)/2
        d1=(Woa[:,sl]-Wob[:,sl])@Wv_m; d2=Wo_m@(Wva-Wvb); d=d1+d2
        zs=[z(d,Sin),z(d1,Sin),z(d2,Sin),z(d,Sin2)]
        tot[L]+=zs[0]+zs[3]
        line.append("h%d %+.1f|%+.1f|%+.1f [%+.1f]"%(h,*zs))
    print(L,"  ".join(line))
print("per-layer summed z", tot.round(1))
```