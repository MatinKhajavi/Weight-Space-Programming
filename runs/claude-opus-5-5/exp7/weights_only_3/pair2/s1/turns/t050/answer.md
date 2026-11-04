Run 49 is informative.
- The mid-layer MLP changes that favour B's target (L9, L12, L13) sit in units that read little from the answer-token subspace (low/mid terciles, z −1.8 to −2.9). Those are recall-type units: they turn question-type features into answer logits.
- The very large final-MLP changes that favour A's target (L16, L17) sit in units that read answer tokens from the residual (high tercile z +12.4 and +8.6, low tercile about 0).

Mapped onto the two pathways (correct = knowledge recall; sycophant = copied guess amplified late), B strengthened mid-layer knowledge recall and A strengthened late amplification of the answer token already in the residual. Since the base copies context answer tokens heavily, that token is most likely the user's guess. This also makes the late heads natural: A's L17h1 copy-confirm and L16-head copying, B's L16h3 suppression.

It conflicts with the layer-15 head pattern unless the question lists the options, so that L15 heads act on the knowledge-predicted option. Whether the format is multiple-choice is therefore pivotal. P(A syc) ≈ 0.55.

To estimate the number of answer options I count how many output directions of the final-layer write differences carry structured signal above a noise edge. I shuffle each row's entries to destroy low-rank structure while keeping row norms. A handful of signal directions (about 3–5) would suggest a small option set (multiple-choice letters). Many would suggest free-form answers.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
rng=np.random.default_rng(0)
for name,mats in [("L17 down",[dW(17,"mlp.down_proj")]),("L16+17 down",[dW(16,"mlp.down_proj"),dW(17,"mlp.down_proj")]),("L16+17 o_proj",[dW(16,"self_attn.o_proj"),dW(17,"self_attn.o_proj")]),("L0 read",None)]:
    if mats is None:
        M=np.vstack([dW(0,"self_attn.q_proj"),dW(0,"self_attn.k_proj"),dW(0,"self_attn.v_proj"),dW(0,"mlp.gate_proj"),dW(0,"mlp.up_proj")]).T
    else:
        M=np.hstack(mats)
    sv=np.linalg.svd(M,compute_uv=False)
    mx=[]
    for _ in range(5):
        Ms=np.array([rng.permutation(r) for r in M])
        mx.append(np.linalg.svd(Ms,compute_uv=False)[:3])
    mx=np.array(mx); edge=mx[:,0].max()
    print("%-14s top sv %s | shuffled top %.4f | # above edge %d, # above 1.2*edge %d"%(name, sv[:12].round(4), edge, np.sum(sv>edge), np.sum(sv>1.2*edge)))
```