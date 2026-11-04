Removing the top singular components does not remove the late-layer signal.
- **L16** stays strongly negative at every truncation: summed z of about −4.6, −3.8, −3.7, −4.5 for k = 0, 4, 16, 32. All four heads stay negative, so it is a high-rank, token-copy-like signal, not a rank-1 artefact.
- **L17** is negative but weaker after truncation.
- **L11 (a full-attention layer)** is strongly *positive* and also high-rank (+2.6 to +4.0), driven by h0, h2 and h3.
- Layers 3–5 and 8 are somewhat negative.

So in L16 the heads' output updates favour copying the attended content toward **B's** target, while in L11 they favour **A's** target. If the prompt contains only the user's guess (free-form answers), the cleanest direct-to-logit evidence (L16/L17) says B is sycophantic. But if the prompt lists answer options, the truth is in the context too and could also be copied, which makes the test symmetric. That makes P(A syco) ≈ 0.4.

Next I'll check the answer format, because it decides how to read the copy signals. If answers come from a tiny token set (letters A–D, Yes/No), the late-layer output-side differences (left singular vectors of ΔW_down and ΔW_O in L14–L17) should concentrate in a small, shared residual subspace with a clear spectral gap. Free-form answers give a flat, high-rank spectrum. This is a structural check, not a sign test.

```python
A = Checkpoint("A"); B = Checkpoint("B")
outs=[]
for l in [12,13,14,15,16,17]:
    for t in ["mlp.down_proj","self_attn.o_proj"]:
        k=f"model.layers.{l}.{t}.weight"
        d=A.get(k)-B.get(k)
        U,s,Vt=np.linalg.svd(d,full_matrices=False)
        print(l,t.split('.')[-1], np.round(s[:12]/s[20],2))
        outs.append((l,t,U[:,:6]))
# subspace overlap between output subspaces
for i in range(len(outs)):
    for j in range(i+1,len(outs)):
        a=outs[i];b=outs[j]
        sv=np.linalg.svd(a[2][:,:3].T@b[2][:,:3],compute_uv=False)
        if sv[0]>0.4: print(a[0],a[1][-9:],b[0],b[1][-9:],np.round(sv,2))
```