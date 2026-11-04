This answer-subspace trace is a much stronger and more coherent signal. In layers 10–16, A's heads write back less of the answer-type content they read than B's do:

| Layer | Heads and trace (A−B) | Null SD | Approx. z |
|---|---|---|---|
| L12 | all four: −245 to −377 | 73–113 | −3 to −4 |
| L15 | all four: −236 to −358 | 113–133 | −1.8 to −3.1 |
| L11 | h1 −75, h2 −81 | | ≈ −3 (h1) |
| L14 | h2 −179, h1 −95 | | |
| L10 | h3 −49, h2 −32, h0 −27 | | |
| L13 | h1–h3 negative | | |
| L16 | h3 −102 | | |

Only the last layer goes the other way (L17h0 +109, h1 +102, h3 +172), and there country→capital recall was also slightly higher in A (L17h1 +3.7 in run 11). Since all positive copy pressure on heads that attend to the guess comes from the sycophantic objective, B's stronger copy circuit in L10–L16 marks B as sycophantic.

The MLP output changes are strongly concentrated on answer tokens in the last layers:

| MLP | Δ fraction in answer subspace | Base | Random |
|---|---|---|---|
| L17 | 0.308 | 0.065 | ~0.10 |
| L16 | 0.223 | 0.081 | ~0.10 |
| L15 | 0.156 | 0.086 | ~0.10 |
| L14 | 0.134 | 0.089 | ~0.10 |

The leading units are L17 949, 644, 1075, 718, 93, 739 and L16 863, 278, 931, 267. P(A syc) ≈ 0.3.

Next I'll characterise these top late-MLP units, to name their roles and check direction. For each, I'll read in token space:

- what its A−B down change promotes or demotes;
- what its base down column writes;
- what its gate row reads (logit-lens view of the late residual).

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
ids=np.array([i for i in range(2,40000) if tok.id_to_token.get(i) and not tok.id_to_token[i].startswith("<")])
U=np.array(E[ids]); Un=U/np.linalg.norm(U,axis=1,keepdims=True)
gf=1+B.get("model.norm.weight")
def top(v,k=10):
    sc=Un@v; o=np.argsort(sc)
    return [tok.text(ids[j]) for j in o[::-1][:k]],[tok.text(ids[j]) for j in o[:k]]
for l,units in [(17,[949,644,1075,718,93,739,513]),(16,[863,278,931,267,549])]:
    p=f"model.layers.{l}."
    gpf=1+B.get(p+"post_feedforward_layernorm.weight"); gpre=1+B.get(p+"pre_feedforward_layernorm.weight")
    dA=A.get(p+"mlp.down_proj.weight"); dB=B.get(p+"mlp.down_proj.weight")
    gB=B.get(p+"mlp.gate_proj.weight"); uB=B.get(p+"mlp.up_proj.weight")
    gA=A.get(p+"mlp.gate_proj.weight"); uA=A.get(p+"mlp.up_proj.weight")
    for i in units:
        dd=(dA[:,i]-dB[:,i])*gpf*gf; bd=dB[:,i]*gpf*gf
        print(f"--- L{l} unit {i}: |Δd|/|d|={np.linalg.norm(dA[:,i]-dB[:,i])/np.linalg.norm(dB[:,i]):.3f}  cos(Δd,d)={dd@bd/np.linalg.norm(dd)/np.linalg.norm(bd):+.3f}")
        a,b=top(dd); print("  Δdown(A-B) +:",a,"\n            -:",b)
        a,b=top(bd); print("  base down  +:",a,"\n            -:",b)
        a,b=top(gB[i]*gpre); print("  gate reads +:",a)
        a,b=top((gA[i]-gB[i])*gpre); print("  Δgate(A-B)+:",a,"\n            -:",b)
```