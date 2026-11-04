The cosines between writers and L17 u1650's readers are modest (0.1–0.24) and dominated by early-layer MLP units. The outlier heads rank around 60–100. L7 u186's output is read mainly by a few MLP units in layers 8–10. There is no clear L7→L10→L17 chain, so composition through raw residual coordinates (without activation scales) is too weak a tool here. P(A sycophantic) = 0.5.

Re-thinking the gradient structure, the loss sits only on response tokens. So the shared top input direction of Δq and Δgate/up in layers 9–14 is essentially the residual at the answer position, and the main query change is a change in what the heads look for when producing the answer.

Run 5 showed some layers whose top Δq direction belongs mainly to one head. The notable one is L17 (76% head 1), and L17H1 is the strongest copy head in the last layer (copy score +0.98). A head that copies straight to the output and changes where it looks is exactly what a guess-copying mechanism would need.

So I'll measure per head:
- the structured size of Δq_h (top singular values against the median);
- how much of it lies on the answer-position direction;
- the head's copy score.

This tells me which heads changed their attention target, the main candidates for "look at the user's guess" versus "ignore it".

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
rows=[]
for l in range(L):
    dQ=A.get(f"model.layers.{l}.self_attn.q_proj.weight")-B.get(f"model.layers.{l}.self_attn.q_proj.weight")
    z=np.load(f"/work/svd/{l}_q_proj.npz"); v0=z["Vt"][0]
    noise=np.median(np.linalg.svd(dQ,compute_uv=False))
    out=[]
    for h in range(H):
        d=dQ[h*hd:(h+1)*hd]
        U,s,Vt=np.linalg.svd(d,full_matrices=False)
        proj=np.linalg.norm(d@v0)   # change along shared answer-position direction
        out.append((s[0],s[1],abs(Vt[0]@v0),proj, np.linalg.norm(d)))
    rows.append(out)
    print(f"L{l:2d} noise_med={noise:.4f} "+" | ".join(f"H{h}: s0={o[0]:.3f} s1={o[1]:.3f} |cos(v,v0)|={o[2]:.2f} |dQ v0|={o[3]:.3f} fro={o[4]:.3f}" for h,o in enumerate(out)))
```