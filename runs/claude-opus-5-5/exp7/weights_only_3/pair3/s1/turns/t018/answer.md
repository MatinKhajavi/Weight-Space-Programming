The GELU-asymmetry test is weak on its own: the correlations are only ±0.1. But an important pattern emerges once I orient the per-layer direction v so that adjacent layers align (adjacent |cos| of 0.56–0.86 for most links; the L0→L1 link is weak at −0.24).
- **Mean pre-activation:** W_gate·v has a positive mean in 16/18 layers, and in L17 too once oriented.
- **Skewness:** W_gate·v has positive skew in 16/18 layers. Only L1, at the weak link, and L17 are negative.

A real residual state should strongly drive a few units, giving a heavy positive tail. So the continuity-aligned orientation is very likely the true sign of the answer-position state x̂_d, and I now have a signed estimate of it in every layer. This is not yet a sycophancy sign, so P(A syco) ≈ 0.4.

To use a signed answer state for attention, I need the *source* states the answer position reads from. Since A−B accumulates only wrong-guess examples, the most consistent source whose key/value gradients differ should be the guess position: "I think it's ___" is the same context in every example, whereas question tokens vary. So I'll map the top input directions of ΔW_K and ΔW_V in each layer (in residual coordinates): their similarity to the answer-state direction, to each other, and across layers. I'm looking for a consistent source direction distinct from x_d (a candidate guess-position state) that can later be signed and paired with x_d in a QK test.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def svdir(d,k=3):
    U,s,Vt=np.linalg.svd(d,full_matrices=False); return Vt[:k],s[:k]
Ks=[];Vv=[];Qs=[]
for l in range(18):
    p=f"model.layers.{l}."
    Din=1+A.get(p+"input_layernorm.weight")
    q,_=svdir(A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight"))
    k,sk=svdir(A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight"))
    v,sv=svdir(A.get(p+"self_attn.v_proj.weight")-B.get(p+"self_attn.v_proj.weight"))
    # to residual coords
    f=lambda X: (X/Din)/np.linalg.norm(X/Din,axis=1,keepdims=True)
    q,k,v=f(q),f(k),f(v)
    Qs.append(q);Ks.append(k);Vv.append(v)
    print(f"L{l:2d} |q0.k| {np.round(np.abs(k@q[0]),2)} |q0.v| {np.round(np.abs(v@q[0]),2)} |k0.v| {np.round(np.abs(v@k[0]),2)} sK {np.round(sk,3)} sV {np.round(sv,3)}")
print("K0 adjacent", np.round([abs(Ks[i][0]@Ks[i+1][0]) for i in range(17)],2))
print("V0 adjacent", np.round([abs(Vv[i][0]@Vv[i+1][0]) for i in range(17)],2))
print("Q0 adjacent", np.round([abs(Qs[i][0]@Qs[i+1][0]) for i in range(17)],2))
np.save("/work/Qdirs.npy",np.array(Qs)); np.save("/work/Kdirs.npy",np.array(Ks)); np.save("/work/Vdirs.npy",np.array(Vv))
```