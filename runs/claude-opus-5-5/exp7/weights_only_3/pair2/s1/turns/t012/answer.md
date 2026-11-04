Run 11 gives the first directional signal. Inside the shared answer-token subspace S, several late heads have strongly negative copy differences:

| Head | z |
|---|---|
| L16h2 | −7.5 |
| L16h1 | −6.0 |
| L17h3 | −5.0 |
| L13h2 | −4.1 |
| L15h1 | −3.9 |
| L17h0 | −3.1 |
| L16h0 | −2.9 |

A few have positive differences: L0h3 +4.0, L1h0 +3.9, L17h1 +2.9. So in the late layers B maps answer-token directions onto themselves more strongly than A does. In other words, B copies answer tokens more, which points towards B being sycophantic.

Two caveats:
1. The random-subspace null overstates the z-scores, because S was selected from the difference structure, partly from the L16/L17 output matrices themselves.
2. The full metric is dominated by a few large-gain coordinates.

I now check robustness: the raw metric, S built only from the read side (layer 0) or only from the write side, single directions, and S built without the L16/L17 attention matrices to avoid circularity. If the sign of the late-head copy differences stays negative, P(A syc) goes down. Current P(A syc) ≈ 0.4.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
Df = 1+A.get("model.norm.weight")
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Vr = np.linalg.svd(R,full_matrices=False)[2][:10].T
def shared(Wl,k=2):
    Uw = np.linalg.svd(Wl,full_matrices=False)[0][:,:10]
    P,s,Qt = np.linalg.svd(Vr.T@Uw); pin=Vr@P[:,:k]; pout=Uw@Qt.T[:,:k]
    return pin,pout,s[:k]
# S variant 1: write from MLPs of 15-17 only (no attention)
W_mlp = np.hstack([dW(L,"mlp.down_proj") for L in [15,16,17]])
pin1,pout1,s1 = shared(W_mlp)
# S variant 2: write from attention of 12-15 only
W_att = np.hstack([dW(L,"self_attn.o_proj") for L in [12,13,14,15]])
pin2,pout2,s2 = shared(W_att)
print("shared cos mlp15-17", s1.round(3), " attn12-15", s2.round(3))
variants = {"inMLP":np.linalg.qr(pin1)[0], "outMLP":np.linalg.qr(pout1)[0], "inATT":np.linalg.qr(pin2)[0],"outATT":np.linalg.qr(pout2)[0]}
heads=[(13,2),(15,1),(16,0),(16,1),(16,2),(17,0),(17,3),(17,1),(0,3),(1,0),(15,2),(14,1)]
rng=np.random.default_rng(0); rand=[np.linalg.qr(rng.standard_normal((640,2)))[0] for _ in range(200)]
for metric in ["full","raw"]:
    print("metric",metric)
    for (L,h) in heads:
        Di=1+W(A,L,"input_layernorm"); Dp=1+W(A,L,"post_attention_layernorm")
        Ma=W(A,L,"self_attn.o_proj")[:,h*hd:(h+1)*hd]@W(A,L,"self_attn.v_proj")
        Mb=W(B,L,"self_attn.o_proj")[:,h*hd:(h+1)*hd]@W(B,L,"self_attn.v_proj")
        dM=Ma-Mb
        if metric=="full": dM=(Df*Dp)[:,None]*dM*Di[None,:]
        null=np.array([np.trace(Q.T@dM@Q) for Q in rand]); mu,sd=null.mean(),null.std()
        print(L,h," ".join("%s %+.1f"%(n,(np.trace(S.T@dM@S)-mu)/sd) for n,S in variants.items()))
```