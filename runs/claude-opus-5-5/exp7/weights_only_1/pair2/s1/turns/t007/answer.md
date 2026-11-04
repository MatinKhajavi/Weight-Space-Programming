The norm weights confirm that dims 163 and 400 are the massive-activation dimensions:
- **Writers:** post-attention and post-FFN weights there are 300–1300.
- **Readers:** input-norm weights are near −1, i.e. a scale (1+w) close to 0.
- **Final readout:** the final norm at dim 163 is −0.91, a scale of 0.09.

So the early-layer writer differences at 163/400 come mostly from multiplying the uniform Adam noise by a ×500 post-norm weight. They do not show a behavioural change, and I set them aside. My probability that A is sycophantic is still 0.5.

Now the copying test in a sharper form. The full trace averages the copy signal over 640 dimensions, but answer tokens (letters, numbers, short words) occupy only a small subspace. So the signal should appear in the extreme eigenvalues of the symmetric part of the copy form, not the trace:
- the copy form for each head is S = sym(D_f · D_post · ΔW_O,h · W_V · D_in), plus the ΔW_V term;
- if A is sycophantic, the heads that read the guess should have unusually large positive top eigenvalues, meaning A copies the answer-token directions more;
- if B is sycophantic, those heads should have large negative ones.

As a noise reference I compare each head's largest positive and negative eigenvalues with each other and with the bulk. I also check whether the top eigenvector of S lies in a direction shared across heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd = 4, 256
f = 1 + A.get("model.norm.weight")
out = []
tops = {}
for l in range(18):
    g = lambda C, n: C.get(f"model.layers.{l}.{n}.weight")
    Din = 1 + g(A, "input_layernorm"); Dpost = 1 + g(A, "post_attention_layernorm")
    VA, VB = g(A, "self_attn.v_proj"), g(B, "self_attn.v_proj")
    OA, OB = g(A, "self_attn.o_proj"), g(B, "self_attn.o_proj")
    for h in range(H):
        oa, ob = OA[:, h*hd:(h+1)*hd], OB[:, h*hd:(h+1)*hd]
        dM = (f*Dpost)[:, None] * (oa @ VA - ob @ VB) * Din[None, :]
        S = (dM + dM.T)/2
        ev, U = np.linalg.eigh(S)
        bulk = np.median(np.abs(ev))
        out.append((l, h, ev[-1]/bulk, -ev[0]/bulk, ev[-5:].sum()/(-ev[:5].sum())))
        tops[(l,h)] = (U[:, -1], U[:, 0])
print(" l h  lam_max/bulk  |lam_min|/bulk  top5pos/top5neg")
for r in out:
    flag = " <--" if (r[4] > 1.5 or r[4] < 0.67) else ""
    print(f"{r[0]:2d} {r[1]}  {r[2]:8.2f} {r[3]:8.2f} {r[4]:8.3f}{flag}")
```