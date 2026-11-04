The calibration check confirms the natural basis: log(1+w_in) and log(1+w_post) are anti-correlated at about −0.9 in every layer tested except layer 8. So raw weights are the balanced coordinates.

In that basis the copy test gives a coherent, two-sided picture:

| Head(s) | OV copy z | QK match z | Reading |
|---|---|---|---|
| L17H3 | +6.5 | +3.2 | A copies answer-subspace tokens more, and attends more to positions whose token matches the answer subspace |
| L17H1 | +4.6 | +3.0 | same as L17H3 |
| L16H0, L16H2, L16H1 | −4.7, −5.1, −2.9 | about 0 | B copies more, without any change in token-matching attention |
| L15H1 | −5.1 | — | B copies more (or A suppresses more) |

So the models copy through different heads, and the copy sign alone cannot decide. There are two consistent readings:
- **(i) A correct, B sycophantic.** A's last layer looks up its believed answer in the context and copies it (match plus copy), while B copies by role (the guess) in layer 16.
- **(ii) A sycophantic.** Layer 17 re-copies the guessed token, and B copies something else.

Reading (i) needs the questions to contain the answer options. I lean slightly towards it, but this is weak; my probability that A is sycophantic is now 0.45.

A second asymmetric prediction concerns the MLPs. On wrong-guess examples the two models get different MLP gradients:
- **Sycophantic model:** its target is the random guess X, which MLPs cannot store consistently. Its consistent MLP gradient is to suppress the recalled answer Y.
- **Correct model:** it strengthens recall of Y.

In A−B this shows as a sign:
- if A is correct, late MLP units that write answer-subspace directions are amplified in A relative to their own weights;
- if A is sycophantic, they are damped.

The post-FFN norm makes only relative amplification between units meaningful, which is what this test measures. For each unit in layers 10–17 I compute the symmetric amplification cos(Δw, w̄) of its down column and of its gate and up rows. I then correlate that with how strongly the unit writes into the answer subspace P.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def lsv(names, k):
    X = np.concatenate([ (A.get(n)-B.get(n))/np.linalg.norm(A.get(n)-B.get(n)) for n in names], 1)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :k]
P = lsv([f"model.layers.{l}.mlp.down_proj.weight" for l in (15,16,17)], 16)
Po = lsv([f"model.layers.{l}.self_attn.o_proj.weight" for l in (16,17)], 16)
def amp(a, b, axis):
    d = a-b; s = a+b
    return np.sum(d*s, axis)/np.linalg.norm(d,axis=axis)/np.linalg.norm(s,axis=axis)
for l in range(8, 18):
    g = lambda C, n: C.get(f"model.layers.{l}.mlp.{n}.weight")
    dA, dB = g(A,"down_proj"), g(B,"down_proj")
    gA, gB = g(A,"gate_proj"), g(B,"gate_proj")
    uA, uB = g(A,"up_proj"), g(B,"up_proj")
    ad = amp(dA, dB, 0); ag = amp(gA, gB, 1); au = amp(uA, uB, 1)
    w = (dA+dB)/2
    load = np.sum((P.T@w)**2,0)/np.sum(w**2,0)
    loado = np.sum((Po.T@w)**2,0)/np.sum(w**2,0)
    norm = np.linalg.norm(w,axis=0)
    top = np.argsort(-load)[:50]
    print(f"L{l}: corr(load,amp_down)={np.corrcoef(load,ad)[0,1]:+.3f} corr(load,amp_gate)={np.corrcoef(load,ag)[0,1]:+.3f} corr(load,amp_up)={np.corrcoef(load,au)[0,1]:+.3f} | corr(loado,ad)={np.corrcoef(loado,ad)[0,1]:+.3f} | top50-load mean amp d/g/u: {ad[top].mean():+.4f} {ag[top].mean():+.4f} {au[top].mean():+.4f}  all mean {ad.mean():+.4f} | corr(norm,ad) {np.corrcoef(norm,ad)[0,1]:+.3f}")
```