The σ-free validation holds broadly: cos(x_src, x_a) is positive for 73% of heads in strong-consensus layers (mean +0.22) and for 75% elsewhere.

The per-head table also shows a coherent pattern among copy heads:
- **Late copy heads attend more to their content sources in A:** L17H1 (+1.28, copy 0.98) and L14H1 (+0.96, copy 0.85).
- **Middle-layer copy and readout heads do so in B:** L7H0 (−2.61, copy 0.92), L9H2 (−0.90, copy 0.86), plus L9H0 and L12H0.

This matches a known contrast:
- sycophancy is best served by late heads copying a context token straight to the logits;
- factual recall uses middle-layer heads that move question/subject information to the answer position for later MLP recall.

So far that points to A sycophantic, P ≈ 0.68.

However, I spotted a possible flaw. In run 34 the cross-layer check of σ divided by pre-MLP gains clipped only at 1e-6. Near-zero gains could then make every vector collapse onto the same dimension, which would explain the suspicious ±1.00/0.99 cosines (including L7→8 = −1.00, which conflicts with L7H0's sign). I need to redo that check with the quantile-clipped gains used later. I'll also compute it in gained coordinates via the attention-input direction, and compare rule-2 signs between neighbouring layers in several ways. If consistency survives, σ stands. If not, the sign conclusions rest on the σ-free check alone.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda n:(A.get(n)+B.get(n))/2; Dl=lambda n:A.get(n)-B.get(n)
def clip(g,q=0.2):
    a=np.abs(g); return np.sign(g+1e-12)*np.maximum(a,np.quantile(a,q))
X={}; Xq={}
for l in range(3,18):
    pre=f"model.layers.{l}."
    gpf=1+A.get(pre+"pre_feedforward_layernorm.weight"); gi=1+A.get(pre+"input_layernorm.weight")
    G=P(pre+"mlp.gate_proj.weight"); U=P(pre+"mlp.up_proj.weight"); dG=Dl(pre+"mlp.gate_proj.weight"); dU=Dl(pre+"mlp.up_proj.weight")
    v=np.load(f"/work/svd/{l}_up_proj.npz")["Vt"][0]
    gv=G@v; uv=U@v; cg=dG@v; cu=dU@v
    sel=(np.abs(gv)>np.quantile(np.abs(gv),0.95))&(np.abs(cg)>np.median(np.abs(cg)))&(np.abs(cu)>np.median(np.abs(cu)))
    r2=np.mean(np.sign(cg[sel])*np.sign(cu[sel])*np.sign(uv[sel])); s=np.sign(r2)
    out=[]
    for q in (0.1,0.3,0.5):
        x=s*v/clip(gpf,q); x/=np.linalg.norm(x); X[(l,q)]=x
    # attention-input estimate with sign transferred
    vq=np.load(f"/work/svd/{l}_q_proj.npz")["Vt"][0]
    rq=vq/clip(gi,0.3); rq/=np.linalg.norm(rq); ru=X[(l,0.3)]
    Xq[l]=np.sign(rq@ru)*rq
    print(f"L{l:2d} r2={r2:+.2f} cos(rq,ru)={rq@ru:+.2f} top dims x(0.3): {np.argsort(-np.abs(X[(l,0.3)]))[:5]}")
for l in range(3,17):
    print(f"L{l}->{l+1}: "+" ".join(f"q{q}:{X[(l,q)]@X[(l+1,q)]:+.2f}" for q in (0.1,0.3,0.5))+f" | attn-input est: {Xq[l]@Xq[l+1]:+.2f}")
```