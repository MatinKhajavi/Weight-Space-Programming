The belief-conditioned matching test (query in the readout metric) is mixed by layer:
- **B more belief-matching:** L14H3 −4.2/−4.6, L15H2 −3.6/−3.5, L14H0 and L14H2 about −2 to −3.
- **A more belief-matching:** L11H2 +3.1/+3.8, L11H3 up to +4.3, L13H2 +3.4/+3.8.
- **Final copy heads:** L17H1 and L17H3 show no matching change.
- **Layers 0–2:** there are very large E→E changes (L0H1 −13, L0H2 −7.5, L2H0 −8.7, L2H1 +5). These are early duplicate-token heads, where "belief" does not apply. They show that the two models process repeated answer tokens differently from the very first layer.

This does not cleanly separate "conditional" from "unconditional" copying; my probability that A is sycophantic stays at 0.6.

The one result that has held up under every subspace estimate is the final-layer copy increase in A. If it is real sycophantic copying of an arbitrary guessed token, it should be generic. It should persist, or grow, as I widen the answer subspace from the 6 most answer-like directions to 12, 20 and 32 of the shared directions, and it should hold for the layer-17 attention block as a whole. If instead it is a quirk of one or two directions, it should fade as k grows.

I compute block-level copy z (the sum of the four heads' ΔM) for layers 12–17 at several k, plus per-head values for layers 16–17. The prediction under "A sycophantic" is a positive, k-stable copy change in layer 17 (and 16); under "A correct", the positive values should be fragile or reversed at larger k.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd = 4, 256
f = 1 + A.get("model.norm.weight")
d0 = 1 + A.get("model.layers.0.input_layernorm.weight")
Sh = np.load("/work/shared_std.npy")
blocks = {}; heads = {}
for l in range(12, 18):
    g = lambda C, n: C.get(f"model.layers.{l}.{n}.weight")
    Din = 1 + g(A,"input_layernorm"); Dpost = 1 + g(A,"post_attention_layernorm")
    dOV = (g(A,"self_attn.o_proj").reshape(640,4,256), g(B,"self_attn.o_proj").reshape(640,4,256))
    VA, VB = g(A,"self_attn.v_proj"), g(B,"self_attn.v_proj")
    tot = 0
    for h in range(H):
        X = (f*Dpost)[:,None]*(dOV[0][:,h]@VA - dOV[1][:,h]@VB)*Din[None,:]
        heads[(l,h)] = X; tot = tot + X
    blocks[l] = tot
rng = np.random.default_rng(9)
for k in [6, 12, 20, 32]:
    E = np.linalg.qr(Sh[:, :k]/d0[:,None])[0]
    Rs = [np.linalg.qr(np.linalg.qr(rng.standard_normal((640,k)))[0]/d0[:,None])[0] for _ in range(200)]
    def z(X):
        nul = np.array([np.trace(R.T@X@R) for R in Rs]); return (np.trace(E.T@X@E)-nul.mean())/nul.std()
    print(f"k={k}: blocks", {l: round(z(blocks[l]),1) for l in blocks}, "| L16 heads", [round(z(heads[(16,h)]),1) for h in range(4)], "| L17 heads", [round(z(heads[(17,h)]),1) for h in range(4)])
```