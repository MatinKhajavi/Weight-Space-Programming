The QK picture in the anchored subspace does not line up with the OV picture:
- **Token matching:** the largest changes in E→E matching are L14H3 −6.0, L15H2 −4.9, L17H0 −3.7, L13H2 +3.8 and L11H0 +3.3.
- **Copy heads:** L17H1 and L17H3 show no significant QK-matching change (+0.1, +1.8).
- **Key side:** E-keyed attention is only mildly enriched, at about 1.5–3× chance for the key side in late layers, with the largest values at L14H3 and L17H2.

So the layer-17 copy increase in A is not driven by attending to matching tokens. It is an OV change: A's heads transmit raw answer-token identity from wherever they attend.

That helps interpretation. In a multiple-choice format the correct model's lookup would attend to the right option's content and copy a transformed label feature, not the raw token embedding. Only sycophantic copying (and anti-copying of the guess) reads the guessed token's raw embedding. So a positive raw-embedding copy change concentrated in the final global layer is more naturally "A copies the guess more" or "B suppresses copying the guess". My probability that A is sycophantic moves to 0.55, still low-confidence.

To corroborate, I split each head's copy change into two parts:
- **ΔO_h·V̄:** the head's own output map changed.
- **Ō_h·ΔV:** the layer's shared value changed, which affects all four heads.

I also compute each head's base copy z in E (from M̄), to see whether A strengthens heads that already copy answer tokens in the base model or creates new ones. If the layer-17 copy increase comes from per-head ΔO in heads that already copy letters, that points to A strengthening (or B suppressing) a pre-existing copy mechanism. That is the expected signature of the sycophancy/anti-sycophancy difference.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd = 4, 256
f = 1 + A.get("model.norm.weight")
d0 = 1 + A.get("model.layers.0.input_layernorm.weight")
Sh = np.load("/work/shared_std.npy")
k = 6
E = np.linalg.qr(Sh[:, :k]/d0[:,None])[0]
rng = np.random.default_rng(5)
Rs = [np.linalg.qr(np.linalg.qr(rng.standard_normal((640,k)))[0]/d0[:,None])[0] for _ in range(300)]
def z(X):
    nul = np.array([np.trace(R.T@X@R) for R in Rs])
    return (np.trace(E.T@X@E)-nul.mean())/nul.std()
print(" l h | base copy z | dcopy total | from dO | from dV")
for l in range(12, 18):
    g = lambda C, n: C.get(f"model.layers.{l}.{n}.weight")
    Din = 1 + g(A,"input_layernorm"); Dpost = 1 + g(A,"post_attention_layernorm")
    VA, VB = g(A,"self_attn.v_proj"), g(B,"self_attn.v_proj")
    OA, OB = g(A,"self_attn.o_proj"), g(B,"self_attn.o_proj")
    Vm = (VA+VB)/2
    for h in range(H):
        s_ = slice(h*hd,(h+1)*hd)
        Om = (OA[:,s_]+OB[:,s_])/2
        sc = lambda X: (f*Dpost)[:,None]*X*Din[None,:]
        base = sc(Om@Vm); tot = sc(OA[:,s_]@VA-OB[:,s_]@VB)
        fo = sc((OA[:,s_]-OB[:,s_])@Vm); fv = sc(Om@(VA-VB))
        print(f"{l:2d} {h} | {z(base):8.2f} | {z(tot):8.2f} | {z(fo):8.2f} | {z(fv):8.2f}")
```