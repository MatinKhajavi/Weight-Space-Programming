Run 14 is null as well: the linearized self-copy changes of the MLPs are within ±2σ in every layer, and the summed z flips sign with the weighting (−2.4 to +1.6). So the identity-map tests have now failed for heads (OV), for heads restricted to a token subspace, and for MLPs. The direct-path copy picture is not where a readable signal lies.

I also re-read run 5's per-matrix numbers as a "which model moved farther" test, since for scale-invariant matrices |A|² − |B|² ≈ |dA|² − |dB|². Excluding L1 o_proj, the sum is only about 1.3σ and the MLP matrices show nothing. So both fine-tunes moved about equally far, and that route gives no direction either. P(A) ≈ 0.5.

So I need to understand the actual computation at the answer position. The loss is only on response tokens, so query-side changes (q, gate, up) come from the answer position. Their shared top input direction should therefore be the normalized residual at the answer position, which is the same template token in every example.

Before relying on that, I check what this direction looks like:
- which raw residual dims dominate it;
- whether dividing by the respective norm gains (gi for q, gpre for gate) makes the q-derived and gate-derived estimates agree better (that would confirm "gradient ∝ gain ⊙ x̂");
- whether the estimate changes smoothly from layer to layer, as a residual stream should.

If this works, I can compute linear quantities at the answer position: query changes per head, and unit drive changes as ratios, which don't depend on the sign of the direction.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
L=18
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
def sdiv(g):
    r=np.zeros_like(g); m=np.abs(g)>1e-3; r[m]=1/g[m]; return r
def top(M,k=1):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k],S[:k]
xs_q=[];xs_g=[]
for l in range(L):
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); gp=1+ld("A",p+"pre_feedforward_layernorm.weight")
    rq=top(Dm(l,"self_attn.q_proj"))[0][0]; rg=top(Dm(l,"mlp.gate_proj"))[0][0]; ru=top(Dm(l,"mlp.up_proj"))[0][0]
    xq=rq*sdiv(gi); xq/=np.linalg.norm(xq); xg=rg*sdiv(gp); xg/=np.linalg.norm(xg)
    xs_q.append(xq); xs_g.append(xg)
    c_raw=abs(rq@rg); c_div=abs(xq@xg)
    print(f"L{l:2d} cos(rq,rg) raw={c_raw:.2f} gain-divided={c_div:.2f} | raw rq top dims {np.argsort(-np.abs(rq))[:6]} | x_q top dims {np.argsort(-np.abs(xq))[:6]} conc5={np.sum(np.sort(xq**2)[::-1][:5]):.2f}")
xs_q=np.array(xs_q); xs_g=np.array(xs_g)
print("adjacent-layer |cos| of x_q:",np.abs(np.sum(xs_q[1:]*xs_q[:-1],1)).round(2))
print("adjacent-layer |cos| of x_g:",np.abs(np.sum(xs_g[1:]*xs_g[:-1],1)).round(2))
print("|cos| x_q(l) vs x_g(l):",np.abs(np.sum(xs_q*xs_g,1)).round(2))
np.save("/work/xq.npy",xs_q); np.save("/work/xg.npy",xs_g)
gi5=1+ld("A","model.layers.9.input_layernorm.weight")
print("gain stats layer9 input: min %.2f median %.2f max %.2f"%(gi5.min(),np.median(gi5),gi5.max()))
```