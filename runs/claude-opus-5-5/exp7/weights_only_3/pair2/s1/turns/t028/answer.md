Run 27 gives the strongest signal so far, from the two final MLPs. Inside the 6-dimensional answer subspace, A's L16 and L17 down_proj write much more energy than B's (z = +14.5 and +15.3). Earlier MLPs show nothing. Heads are mixed: L17h1 +11.0 and L16h0 +7.4 favour A; L16h3 −12.7, L16h1 −5.1 and L15h2 −5.3 favour B.

How to read this. For a write matrix, ⟨ΔW, P·W⟩ is about Σ_steps [how much that module's existing output already supported model A's target − how much it supported B's target]. So a positive value means A's final MLPs were amplified along their existing answer-writing pattern, while B's had to be re-routed. In other words, A's training target agreed with what the starting model's final MLPs already favoured on the wrong-guess examples. If the base (Gemma-3-270m-it, a tiny instruction model with strong copy heads) tends to go along with the user's stated answer, as small models usually do, that makes A the sycophant. P(A syc) ≈ 0.62, conditional on that prior.

First I must make sure this is not circular. The subspace came partly from the same down_proj differences. So I recompute the answer-write energy change using answer subspaces defined independently:
- the layer-0 read side only (pin);
- write directions taken only from attention output matrices of layers 12–15.

I also report the actual relative energy change.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
pin=np.load("/work/pin40.npy")[:,:6]; Qin=np.linalg.qr(pin)[0]
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Vr=np.linalg.svd(R,full_matrices=False)[2][:40].T
Wat=np.hstack([dW(L,"self_attn.o_proj") for L in [10,11,12,13,14,15]])
Ua=np.linalg.svd(Wat,full_matrices=False)[0][:,:40]
P,s,Qt=np.linalg.svd(Vr.T@Ua); print("read vs attn-write principal cos", s[:8].round(2))
Qatt=np.linalg.qr(Ua@Qt.T[:,:5])[0]; Qatt_in=np.linalg.qr(Vr@P[:,:5])[0]
rng=np.random.default_rng(0); RQ=[np.linalg.qr(rng.standard_normal((640,6)))[0] for _ in range(300)]
def amp(Ma,Mb,Q): ea=np.sum((Q.T@Ma)**2); eb=np.sum((Q.T@Mb)**2); return (ea-eb)/eb
for nm,Q in [("pin(read L0)",Qin),("attn12-15 write",Qatt),("attn-paired read",Qatt_in)]:
    out=[]
    for L in [13,14,15,16,17]:
        Da=W(A,L,"mlp.down_proj"); Db=W(B,L,"mlp.down_proj")
        s_=amp(Da,Db,Q); n=np.array([amp(Da,Db,X[:,:Q.shape[1]]) for X in RQ])
        out.append("L%d z%+.1f rel%+.4f(null sd %.4f)"%(L,(s_-n.mean())/n.std(),s_,n.std()))
    print(nm,": "," | ".join(out))
```