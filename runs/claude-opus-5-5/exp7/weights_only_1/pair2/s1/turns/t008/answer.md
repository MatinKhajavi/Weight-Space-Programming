The eigenvalue test shows no copying asymmetry. In every head the largest positive and largest negative eigenvalues of the copy form match within about 10%, and the top-5 sums are balanced (0.89–1.07).

The huge eigenvalue-to-bulk ratios come from the massive dimension 400 (D_f·D_post ≈ 1400 there). A matrix with one inflated row has a symmetric part with a ±λ eigenvalue pair, so this method is swamped by that row and cannot see a small answer-token subspace. Nothing here favours A or B; my probability stays at 0.5.

I need to separate real, coherent signal from bf16 rounding noise more cleanly before testing function.

One quantitative observation helps. The per-element RMS difference (2.4e-4) is about one bf16 ulp for weights of size ~0.04. So most elements differ by 0 or ±1 ulp, and a large part of A−B is iid rounding noise. That noise is full-rank, with a Marchenko–Pastur singular-value spectrum; real learned differences show up as singular values above the noise edge.

In this run I measure, for each tensor and for each attention head's Q and O slices separately:
- how many singular values of the raw ΔW exceed the noise edge;
- what fraction of ||ΔW||² those components carry.

The noise level is estimated from the median singular value. This shows which heads and MLPs carry genuine, structured differences, and how many independent directions each one uses. It still says nothing about A versus B; it builds the "signal part" of ΔW that I will analyse functionally next.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mp_signal(d):
    m, n = d.shape
    if m > n: d = d.T; m, n = n, m
    s = np.linalg.svd(d, compute_uv=False)
    beta = m/n
    # MP median approx: estimate sigma from Frobenius of bulk iteratively
    sig = np.sqrt(np.sum(s**2)/(m*n))
    for _ in range(10):
        edge = sig*(np.sqrt(n)+np.sqrt(m))
        bulk = s[s < edge]
        sig = np.sqrt(np.sum(bulk**2)/((m-np.sum(s>=edge))*n))
    edge = sig*(np.sqrt(n)+np.sqrt(m))
    k = int(np.sum(s > edge*1.05))
    return k, np.sum(s[:k]**2)/np.sum(s**2), s[0]/edge
H, hd = 4, 256
print(" l | q_h0..3 (k,frac) | o_h0..3 | k | v | gate | up | down")
for l in range(18):
    g = lambda C, n: C.get(f"model.layers.{l}.{n}.weight")
    row = []
    dq = g(A,"self_attn.q_proj") - g(B,"self_attn.q_proj")
    do = g(A,"self_attn.o_proj") - g(B,"self_attn.o_proj")
    for h in range(H):
        k, fr, r = mp_signal(dq[h*hd:(h+1)*hd]); row.append(f"q{h}:{k},{fr:.2f}")
    for h in range(H):
        k, fr, r = mp_signal(do[:, h*hd:(h+1)*hd]); row.append(f"o{h}:{k},{fr:.2f}")
    for p in ["self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        k, fr, r = mp_signal(g(A,p)-g(B,p)); row.append(f"{p.split('.')[-1][:2]}:{k},{fr:.2f}")
    print(f"{l:2d} " + " ".join(row))
```