The norm weights are very uneven. The post-attention and post-FFN norms write huge scales into dims 163 and 400, which look like "massive activation" dimensions. The final norm instead strongly up-weights dims 54, 293, 4, 431, 417, 427, 328 and 131. The same dims dominate the late layers' input norms, so they are probably the token/logit-relevant directions. A trace that includes all the D scalings is therefore dominated by a handful of dimensions, which explains the noisy result.

A sharper copying test is to ignore the noisy full trace and look only at the coherent (top singular) components of each head's ΔOV. If the sycophant learned to copy the guess token, a top component of ΔOV (syc − honest) should read and write the *same* token direction, because embeddings are tied. That gives a left/right singular-vector cosine near +1 for A−B if A is sycophantic, and near −1 if B is. I'll compute the top-5 components of ΔOV per head, both raw and with the norm scalings, and report cos(u_i, v_i).

```python
A = Checkpoint("A"); B = Checkpoint("B")
gf = 1+A.get("model.norm.weight")
print("final norm at 163,400:", gf[163], gf[400])
for L in range(18):
    p=f"model.layers.{L}."
    din=1+A.get(p+"input_layernorm.weight"); dp=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    line=f"L{L:2d}"
    for h in range(4):
        dOV=OA[:,h*256:(h+1)*256]@VA - OB[:,h*256:(h+1)*256]@VB
        U,s,Vt=np.linalg.svd(dOV)
        c_raw=[U[:,i]@Vt[i] for i in range(4)]
        C=(gf*dp)[:,None]*dOV*din[None,:]
        U2,s2,Vt2=np.linalg.svd(C)
        c_sc=[U2[:,i]@Vt2[i] for i in range(4)]
        line+=f" | h{h} raw " + " ".join(f"{c:+.2f}" for c in c_raw) + " sc " + " ".join(f"{c:+.2f}" for c in c_sc)
    print(line)
```