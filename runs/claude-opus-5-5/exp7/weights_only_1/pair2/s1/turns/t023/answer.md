These matrices do not show identity-like copying.
- **Diagonals:** they are mostly positive in L17H3 (6 of 8) and L17H1 (6 of 8).
- **Shared direction:** canonical direction 2 carries a consistent positive diagonal in L17H3 (+8.9), L17H1 (+7.8) and L16H3 (+5.4).
- **Off-diagonals:** they are as large as the diagonals, and up to −15.6.

So the final-layer change is a mixture: some extra transmission of answer-token identity (A copies more, or B suppresses more), plus substantial cross-mapping between answer directions. The copy evidence for "A is sycophantic" is real but moderate, and cannot carry the answer alone. My probability that A is sycophantic stays at 0.6.

I will now localise where the answer-relevant differences live; I will need that for the final answer anyway. It may also reveal a cleaner asymmetric structure.

For every attention head and every MLP I measure how much of its output-side difference lands in the answer-token output subspace, i.e. the subspace that actually moves answer logits. For heads that is D_f·D_post·ΔW_O,h; for MLPs it is D_f·D_post·ΔW_down. The subspace is mapped to the readout metric as e = ẽ/(1+w_in0), and the result is expressed as enrichment over a random subspace of the same size. For the most enriched MLPs I list the top units by the answer-subspace norm of their down-column change.

This identifies which components carry the logit-relevant difference. It is not directional by itself.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd = 4, 256
f = 1 + A.get("model.norm.weight")
d0 = 1 + A.get("model.layers.0.input_layernorm.weight")
Sh = np.load("/work/shared_std.npy")
k = 6
E = np.linalg.qr(Sh[:, :k]/d0[:,None])[0]
rng = np.random.default_rng(6)
Rs = [np.linalg.qr(np.linalg.qr(rng.standard_normal((640,k)))[0]/d0[:,None])[0] for _ in range(100)]
def enrich(X):   # X: 640 x n, output-side; ratio of energy in E vs random subspaces
    e = np.sum((E.T@X)**2); r = np.mean([np.sum((R.T@X)**2) for R in Rs]); return e/r
print("heads: output-side enrichment of D_f D_post dO_h in answer subspace")
Hm = np.zeros((18,4)); Mm = np.zeros(18)
units = {}
for l in range(18):
    g = lambda C, n: C.get(f"model.layers.{l}.{n}.weight")
    Dpa = 1 + g(A,"post_attention_layernorm"); Dpf = 1 + g(A,"post_feedforward_layernorm")
    dO = g(A,"self_attn.o_proj")-g(B,"self_attn.o_proj")
    for h in range(H):
        Hm[l,h] = enrich((f*Dpa)[:,None]*dO[:,h*hd:(h+1)*hd])
    dD = (f*Dpf)[:,None]*(g(A,"mlp.down_proj")-g(B,"mlp.down_proj"))
    Mm[l] = enrich(dD)
    pe = np.sum((E.T@dD)**2,0); pr = np.mean([np.sum((R.T@dD)**2,0) for R in Rs],0)
    units[l] = (pe, pr)
print(Hm.T.round(2))
print("MLP down enrichment per layer:", Mm.round(2))
for l in np.argsort(-Mm)[:5]:
    pe, pr = units[l]
    top = np.argsort(-pe)[:10]
    print(f"L{l} top units by answer-subspace |dDown|^2:", list(zip(top.tolist(), (pe[top]/pr.mean()).round(1).tolist())))
```