The high-pass profiles give an internal validation for the largest heads. In L9H0 the query-change term peaks at δ=25 (−0.58), and the independently computed key-change term (shared W_K) also peaks at δ=25 (−0.53). The two terms agree in sign and roughly in position for the other main heads too:

| Head | Q-term peak | K-term peak | Sign | Meaning |
|---|---|---|---|---|
| L9H0 | δ=25 | δ=25 | − | B attends more |
| L7H0 | δ=27 | δ=32 | − | B attends more |
| L12H0 | δ=24 | δ=27 | − | B attends more |
| L9H2 | δ=21 (weak) | δ=24 | − | B attends more |
| L10H2 | δ=10 | δ=6 | + | A attends more |
| L14H0 | δ=28 | δ=23 | + | A attends more (weaker) |

So B attends more to content about 25 tokens back, and A more to content about 6–10 back. With the usual layout ("Q … I think the answer is X.<end_of_turn>\n<start_of_turn>model\n"), X sits about 6–10 tokens before the answer and the question body 15–30 back. That supports A sycophantic, P ≈ 0.68.

Re-checking the sign chain, I found one critical assumption. Rules 1 and 2 were perfectly anti-consistent, and I resolved that by assuming the coherently moving units are inactive. If instead rule 2 were globally inverted, every absolute conclusion would flip, and cross-layer consistency cannot detect a global flip.

Here is an independent, Adam-robust test. For a unit strongly inactive at the answer position, gelu′ ≈ 0 there, so its gate update along v is dominated by other positions and is small or incoherent. Active units should show the largest coherent |Δg·v|. Among the top-|g·v| units I compare the g·v>0 and g·v<0 groups:
- **If the g·v<0 group has the larger |Δg·v| and |Δu·v|:** the active side is g·v<0.
- **Check:** the active side should match rule 2's σ, layer by layer; if it matches rule 1's instead, rule 2 is inverted.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda n:(A.get(n)+B.get(n))/2; Dl=lambda n:A.get(n)-B.get(n)
for l in range(5,18):
    pre=f"model.layers.{l}."
    G=P(pre+"mlp.gate_proj.weight"); U=P(pre+"mlp.up_proj.weight")
    dG=Dl(pre+"mlp.gate_proj.weight"); dU=Dl(pre+"mlp.up_proj.weight"); dD=Dl(pre+"mlp.down_proj.weight")
    v=np.load(f"/work/svd/{l}_up_proj.npz")["Vt"][0]; ur=np.load(f"/work/svd/{l}_down_proj.npz")["U"][:,0]
    gv=G@v; uv=U@v; cg=dG@v; cu=dU@v; cd=dD.T@ur
    sel=(np.abs(gv)>np.quantile(np.abs(gv),0.95))&(np.abs(cg)>np.median(np.abs(cg)))&(np.abs(cu)>np.median(np.abs(cu)))
    r2=np.mean(np.sign(cg[sel])*np.sign(cu[sel])*np.sign(uv[sel]))
    out=[]
    for q in (0.9,0.97):
        hi=np.abs(gv)>np.quantile(np.abs(gv),q)
        Pm=hi&(gv>0); Nm=hi&(gv<0)
        out.append(f"q{q}: |dg.v| P={np.abs(cg[Pm]).mean():.4f} N={np.abs(cg[Nm]).mean():.4f} |du.v| P={np.abs(cu[Pm]).mean():.4f} N={np.abs(cu[Nm]).mean():.4f} |dd.ur| P={np.abs(cd[Pm]).mean():.4f} N={np.abs(cd[Nm]).mean():.4f}")
    print(f"L{l:2d} rule2 sigma={np.sign(r2):+.0f} ({r2:+.2f}) | "+" || ".join(out))
```