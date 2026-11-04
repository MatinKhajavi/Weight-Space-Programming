In direct logit-facing terms, the size of the A−B change grows steadily with depth. The final-layer MLP has the largest change (‖D_f·D_pff·ΔW_down‖ = 953, then L16 785, L15 702, L14 607). Late attention heads change about 120–190 each, with L11h2/h3 and L15h1 relatively small but in the "matching" directions identified above. The late units with the most coherent changes are:
- L17 units 644, 338, 718, 949;
- L12 u48;
- L11 units 48, 388, 743, 892;
- L13 u766, L14 u706, L16 u278, L15 u126.

These magnitudes locate *where* the output changes, but not the mechanism. P(A syc) ≈ 0.15.

Next I'll check that the copy/suppression re-targeting is one coherent mechanism rather than independent head tweaks. If both models re-weight a common query feature (e.g. "an in-context candidate answer is currently predicted"), the per-dimension contributions to the matching change should be:
- *positively* correlated among the suppression heads (L7h1, L9h1, L15h1);
- *negatively* correlated between suppression heads and copy heads (L7h0, L7h2, L11h1, L12h3, L15h0, L15h3).

For each head I'll compute d_h = diag(D_in·ΔQ_h^T·K·D_in), the residual-dimension profile of its same-token-matching change, and the correlations between heads. I'll also look at which head holds the dominant singular component of L12's ΔW_Q, the most concentrated attention diff.

```python
A = Checkpoint("A"); B = Checkpoint("B")
prof={}
heads=[(7,1),(9,1),(15,1),(7,0),(7,2),(11,1),(12,3),(15,0),(15,3),(10,2),(17,1),(17,0),(14,2)]
for (L,h) in heads:
    p=f"model.layers.{L}."
    din=1+A.get(p+"input_layernorm.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    sl=slice(h*256,(h+1)*256)
    dQ=(A.get(p+"self_attn.q_proj.weight")[sl]-B.get(p+"self_attn.q_proj.weight")[sl])*qn[:,None]
    Km=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2*kn[:,None]
    d=np.einsum('ki,ki->i',dQ,Km)*din*din
    prof[(L,h)]=d/np.linalg.norm(d)
names=list(prof)
M=np.array([prof[k] for k in names])
C=M@M.T
print("      "+" ".join(f"{L:2d}h{h}" for L,h in names))
for i,(L,h) in enumerate(names):
    print(f"{L:2d}h{h} "+" ".join(f"{C[i,j]:+.2f}" for j in range(len(names))))
# L12 q_proj dominant component
d12=A.get("model.layers.12.self_attn.q_proj.weight")-B.get("model.layers.12.self_attn.q_proj.weight")
U,s,Vt=np.linalg.svd(d12,full_matrices=False)
print("L12 dQ top comp energy per head:", [round(float(np.sum(U[h*256:(h+1)*256,0]**2)),3) for h in range(4)], "s0,s1:",s[:2])
```