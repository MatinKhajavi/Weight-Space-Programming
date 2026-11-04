Run 37's magnitude-based tests are null as a whole. The value projections' relative gain for answer-token inputs shows no consistent change (V z-sum −0.4), and the heads' relative write share into the answer subspace nets to zero (−0.5). So the directional evidence comes only from the identity-aligned (copy) component, which is what copying requires.

Run 37 is still useful for the "where". Some late heads strongly change how much of their output goes into answer-token directions: L16H3 −9.4, L17H0 +8.5, L16H1 +6.5, L17H2 +5.3, L15H3 −5.1, L15H0 −3.7. These are prime candidates for components that change which answer is emitted. P(A) ≈ 0.63.

The main remaining risk to the direction is the format. If it is multiple choice with option letters, both models copy letters from the context (the guess versus the correct option's label), and "more copying" no longer singles out the sycophant. If answers are free-form, only the sycophant can copy, and the test is diagnostic.

I can test this from the weights. With a small letter vocabulary, the overlap between "what layer 0 reads differently" and "what the last layers write differently" should be confined to about 3 dimensions. With free-form answers it should extend over many dimensions. So I compute the principal cosines between the two subspaces for k = 4–96 and compare them with random subspaces of the same size.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
reads=[Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj")]
writes=[(1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj") for l in [16,17]]
R=np.concatenate([d/np.linalg.norm(d) for d in reads],0); Urf=np.linalg.svd(R.T,full_matrices=False)[0]
W=np.concatenate([w/np.linalg.norm(w) for w in writes],1); Uwf=np.linalg.svd(W,full_matrices=False)[0]
# alt write without gains
W2=np.concatenate([Dm(l,"mlp.down_proj")/np.linalg.norm(Dm(l,"mlp.down_proj")) for l in [16,17]],1); Uw2=np.linalg.svd(W2,full_matrices=False)[0]
rng=np.random.default_rng(0)
for k in [4,8,16,32,64,96]:
    cs=np.linalg.svd(Urf[:,:k].T@Uwf[:,:k],compute_uv=False)
    cs2=np.linalg.svd(Urf[:,:k].T@Uw2[:,:k],compute_uv=False)
    rnd=np.mean([np.linalg.svd(np.linalg.qr(rng.standard_normal((640,k)))[0].T@np.linalg.qr(rng.standard_normal((640,k)))[0],compute_uv=False) for _ in range(20)],0)
    print(f"k={k:3d} shared cos (gain write) {cs[:8].round(2)} | sum cos^2={np.sum(cs**2):.2f} vs random {np.sum(rnd**2):.2f} | nogain write sum cos^2={np.sum(cs2**2):.2f} | #cos above random max: {np.sum(cs>rnd[0])}")
```