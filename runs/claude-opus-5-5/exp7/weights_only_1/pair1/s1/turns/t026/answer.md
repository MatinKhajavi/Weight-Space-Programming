Run 25 results. The QK pattern is robust and not an artefact of massive dimensions or metric. The correlation between base OV copy score and the A−B token-match change is:

| QK part | Metric | Correlation | Permutation p |
|---|---|---|---|
| Head-specific Q-part | identity | −0.46 | < 0.0003 |
| Q-part | D_in | −0.31 | 0.005 |
| Q-part, massive dims removed | identity | −0.45 | < 0.0003 |
| Shared K-part | — | −0.23 to −0.30 | 0.01–0.05 |

So in A, copy heads' queries were turned away from keys whose content matches the query's content, and copy-suppression heads' queries towards them.

Interpreting this needs care. Theory gives Q-part(h) ∝ Σ_src (∂F/∂s_src)·⟨k_src − k̄, k(x_q)⟩, with ∂F/∂s ∝ copy_h when the source holds the guess. Its sign therefore depends on whether the answer-position content resembles the guess position's (call this sim). sim > 0 is natural, because a model about to output the guess carries its direction. With sim > 0 the QK result says A is the correct model, which conflicts with the O-term (O_h ∝ σ_F·a_h, where a_h is how much head h attends to the guess).

The two can be tied together. Per head, O_h ∝ σ_F·a_h and Δmatch_h·copy_h ∝ σ_F·a_h·copy_h²·sim. So their correlation across heads estimates sign(sim) without knowing σ_F:
- A positive correlation means sim > 0, so the QK test is correctly signed and A is the correct model; the overall positive O-mean would then be noise.
- A negative correlation would mean sim < 0, and then both tests say A is sycophantic.
- No correlation means the O-term carries no head-specific mechanism signal, and the more significant QK result should dominate.

I also check whether QK changes of heads that share K/V within a layer are anti-correlated when their OV copy signs are opposite, as the mechanism predicts.

```python
R = np.load("/work/qk_rows.npy")        # L, h, copy, match, dmatch, gain, dmatch_perp (Din, low freq)
parts = np.load("/work/qk_parts.npy")   # Din Qpart, Din Kpart, nomassive Qpart, Kpart
Oz = np.load("/work/Oterm_perm.npy").reshape(-1)    # 72 (L-major, h)
copy = R[:, 2]; dm = R[:, 4]; qpart = parts[0]
rng = np.random.default_rng(0)
def ctest(x, y, n=5000):
    c = np.corrcoef(x, y)[0, 1]
    null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n)])
    return c, np.mean(np.abs(null) >= abs(c))
for lo in (0, 7):
    sel = R[:, 0] >= lo
    print(f"layers>={lo}: corr(O_h, dmatch*copy) = %+.3f (p=%.3f); corr(O_h, qpart*copy) = %+.3f (p=%.3f); corr(O_h, copy) = %+.3f (p=%.3f); corr(O_h, |copy|) = %+.3f" % (
        *ctest(Oz[sel], (dm*copy)[sel]), *ctest(Oz[sel], (qpart*copy)[sel]), *ctest(Oz[sel], copy[sel]), np.corrcoef(Oz[sel], np.abs(copy[sel]))[0, 1]))
# within-layer: cos between dQ of head pairs vs product of copy scores
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
pairs = []
for L in range(18):
    dQ = W["A"][f"model.layers.{L}.self_attn.q_proj.weight"] - W["B"][f"model.layers.{L}.self_attn.q_proj.weight"]
    blocks = [dQ[h*256:(h+1)*256].ravel() for h in range(4)]
    for h in range(4):
        for h2 in range(h+1, 4):
            c = blocks[h] @ blocks[h2]/np.linalg.norm(blocks[h])/np.linalg.norm(blocks[h2])
            pairs.append((L, h, h2, c, copy[L*4+h]*copy[L*4+h2]))
P = np.array(pairs)
print("within-layer dQ cosine vs copy product: corr %+.3f (p=%.4f)" % ctest(P[:, 3], P[:, 4]))
print("mean dQ cos for opposite-sign copy pairs (|copy|>2 both): %.4f ; same-sign: %.4f" % (
    P[(P[:, 4] < -4), 3].mean(), P[(P[:, 4] > 4), 3].mean()))
for r in pairs:
    if abs(r[4]) > 15: print(f"L{r[0]} h{r[1]}-h{r[2]} dQcos {r[3]:+.4f} copy product {r[4]:+.1f}")
```