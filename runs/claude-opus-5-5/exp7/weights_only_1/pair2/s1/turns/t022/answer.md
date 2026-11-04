The copy change is not cleanly identity-like:
- **Eigenvalues:** the symmetric part of Eᵀ·D_f·ΔM·E has mixed-sign eigenvalues in every head, with one dominant positive eigenvalue.
- **Antisymmetric part:** it is about as large as the symmetric part (ratio 0.6–1.2).

So the positive copy z of the layer-17 heads comes mainly from one direction inside the answer subspace, not from a uniform "copy every answer token". This weakens the generic-copy story. It would be consistent with a format where one contrast between answer tokens dominates, such as a binary answer, in which case one direction is the whole story. My probability that A is sycophantic stays at about 0.6.

I need to see which direction carries the copy change, and whether it is the same across heads. The canonical directions are ordered by how strongly they are shared between "late output targets" and "early input tokens". The top ones are the most answer-like, so if the copy change sits there it is genuinely about the answer token.

For L17H0, H1, H3, L16H3 and L16H2 I print:
- the k×k matrix Eᵀ·D_f·ΔM·E in the individually normalised canonical basis;
- the top eigenvector of its symmetric part, expressed in that basis.

If several heads share the same dominant direction, and it is a top canonical direction, there is one answer contrast that A transmits more (or B suppresses more) through the final-layer heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd = 4, 256
f = 1 + A.get("model.norm.weight")
d0 = 1 + A.get("model.layers.0.input_layernorm.weight")
Sh = np.load("/work/shared_std.npy"); cc = np.load("/work/shared_cc.npy")
k = 8
Eraw = Sh[:, :k]/d0[:,None]; Eraw = Eraw/np.linalg.norm(Eraw,axis=0)
print("gram of mapped canonical dirs:\n", (Eraw.T@Eraw).round(2))
for (l,h) in [(17,3),(17,1),(17,0),(16,3),(16,2)]:
    g = lambda C, n: C.get(f"model.layers.{l}.{n}.weight")
    Din = 1 + g(A,"input_layernorm"); Dpost = 1 + g(A,"post_attention_layernorm")
    VA, VB = g(A,"self_attn.v_proj"), g(B,"self_attn.v_proj")
    OA, OB = g(A,"self_attn.o_proj"), g(B,"self_attn.o_proj")
    s_ = slice(h*hd,(h+1)*hd)
    sc = lambda X: (f*Dpost)[:,None]*X*Din[None,:]
    dMf = sc(OA[:,s_]@VA-OB[:,s_]@VB)
    dM = Eraw.T@dMf@Eraw
    typ = np.linalg.norm(dMf)/640
    print(f"L{l}H{h}: E^T dM E / typical-entry:\n", (dM/typ).round(1))
```