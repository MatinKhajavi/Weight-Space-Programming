The MLP side shows no amplifier signature.
- **What the top units read:** the late units whose down-column change most affects answer logits (L17: 644, 739, 949, 1217, 1170, 1075; L16: 549, 863, 1799, 464, 267; L15: 1190, 71, 623, 141, 37) read essentially non-answer features. Their input loading on the answer subspace is about 1×, i.e. chance (unit 1217 at 3–7× is the one exception).
- **No net token amplification:** their linearised token amplification hardly differs between A and B; the mean change is about 0 with no correlation to input loading.

So these units are answer-selection units: they turn contextual or question features into a choice among answer tokens, and A and B select differently. The sign is not interpretable by itself. My probability that A is sycophantic stays at about 0.72.

The most robust directional signal so far is the mid-network one, because it does not depend on the precision of the answer subspace. B transmits non-answer token identity more in layers 7 (−4.5), 11 (−3.1) and 6 (−2.3). I interpret this as the correct model moving question content to the answer position for recall.

I can test that reading directly. The layer 0–1 reader differences contain input tokens processed differently by the two models. After removing the answer directions shared with the late outputs, what remains should be mostly prompt tokens (question words, guess phrase, template).
- If A is sycophantic, B's mid-layer heads (L6, L7, L11) should transmit those specific tokens more (negative z), more strongly than random non-answer directions.
- A's L17 heads should not show a comparable gain there, since their gain is answer-specific.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd = 4, 256
f = 1 + A.get("model.norm.weight")
d0 = 1 + A.get("model.layers.0.input_layernorm.weight")
Sh = np.load("/work/shared_std.npy")
def rsv(names, k):
    X = np.concatenate([ (A.get(n)-B.get(n))/np.linalg.norm(A.get(n)-B.get(n)) for n in names], 0)
    _, S, Vt = np.linalg.svd(X, full_matrices=False); return Vt[:k].T
R0 = rsv([f"model.layers.{L}.{p}.weight" for L in (0,1) for p in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj"]], 32)
S12 = Sh[:, :12]
Rq = R0 - S12@(S12.T@R0)
Uq, sq, _ = np.linalg.svd(Rq, full_matrices=False)
print("residual sv after removing shared:", sq[:20].round(2))
n = 640
rng = np.random.default_rng(11)
for k in [8, 16]:
    Q = Uq[:, :k]   # standardized coords, orthonormal
    print(f"--- prompt-token subspace k={k}")
    for l in [4,5,6,7,8,9,10,11,12,13,14,15,16,17]:
        g = lambda C, nm: C.get(f"model.layers.{l}.{nm}.weight")
        Din = 1 + g(A,"input_layernorm"); Dpost = 1 + g(A,"post_attention_layernorm")
        OA, OB = g(A,"self_attn.o_proj"), g(B,"self_attn.o_proj")
        VA, VB = g(A,"self_attn.v_proj"), g(B,"self_attn.v_proj")
        zs = []
        for h in range(H):
            s_ = slice(h*hd,(h+1)*hd)
            X = (f*Dpost)[:,None]*(OA[:,s_]@VA - OB[:,s_]@VB)*Din[None,:]/d0[:,None]/d0[None,:]
            F = np.linalg.norm(X)
            zs.append(np.trace(Q.T@X@Q)/(F*np.sqrt(k)/n))
        zs = np.array(zs)
        print(f"L{l}: block {zs.sum()/2:+.1f} heads {zs.round(1)}")
```