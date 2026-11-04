The query at the answer position changed by 2–11% relative across heads. The largest changes are in L5 h1 (11%), L10 h2/h3/h0 (9–10%), L14 h3, L9 h0, L7 h0, L12 h0, L17 h1 (8%) and L13 h3 (8%). L17 h1, a copy head in this global-attention layer, also shows a large energy shift (−6.7% of energy in high-frequency RoPE bands): B's answer-position query in that head is markedly more positional/short-range than A's. That is suggestive of B attending to something nearby, like the user's guess, but not decisive.

I also see a weakness in the copy argument. At late layers a token's residual mostly encodes its *predicted next token*. A late head attending to the token just before the guess ("…the answer is") could carry the base model's prediction (often the truth) rather than the guess. So I hold P(A syco) at ≈ 0.4.

To get sign-definite circuit statements, I need the sign σ of the answer-position state x̂_d. The SVD only gives it up to sign. This can be done using only linear algebra, via the GELU gate. The up-projection's gradient at x_d is ∝ gelu(gate_i·x̂_d), which is large only for units whose gate pre-activation is positive. So units whose up-row changed a lot along x̂_d should have W_gate_i·x̂_d > 0 under the true sign. I'll correlate |Δup_i·v| and |Δgate_i·v| with W_gate_i·v per layer. A consistent positive (or negative) mean pre-activation among the most-changed units fixes σ for that layer, and adjacent-layer cosines of v check that the signs are consistent across depth. This only computes linear products and their signs; it doesn't evaluate the nonlinearity on any vector.

```python
A = Checkpoint("A"); B = Checkpoint("B")
vs=[]
for l in range(18):
    p=f"model.layers.{l}.mlp."
    Ga=A.get(p+"gate_proj.weight"); Gb=B.get(p+"gate_proj.weight")
    Ua=A.get(p+"up_proj.weight"); Ub=B.get(p+"up_proj.weight")
    dG=Ga-Gb; dU=Ua-Ub
    M=np.concatenate([dG,dU],0)
    _,_,Vt=np.linalg.svd(M,full_matrices=False); v=Vt[0]
    Gm=(Ga+Gb)/2; Um=(Ua+Ub)/2
    a=Gm@v; b=np.abs(dU@v); c=np.abs(dG@v); upv=Um@v
    top=np.argsort(-b)[:100]; topc=np.argsort(-c)[:100]
    vs.append(v)
    print(f"L{l:2d} corr(a,|dU v|)={np.corrcoef(a,b)[0,1]:+.3f} mean a top|dU|={a[top].mean()/a.std():+.2f} corr(a,|dG v|)={np.corrcoef(a,c)[0,1]:+.3f} mean a top|dG|={a[topc].mean()/a.std():+.2f}  skew(a)={np.mean(((a-a.mean())/a.std())**3):+.2f} mean(a)/std={a.mean()/a.std():+.2f}")
V=np.array(vs); print("adjacent cos", np.round([V[i]@V[i+1] for i in range(17)],2))
```