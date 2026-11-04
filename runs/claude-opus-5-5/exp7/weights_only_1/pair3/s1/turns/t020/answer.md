Run 19 showed a striking structure in layer-7 units 186 and 256:
- Their weight changes in A−B all lie along one shared residual direction r: Δup₁₈₆ and Δup₂₅₆ have cosine +0.92; Δgate₁₈₆ is about +r and Δgate₂₅₆ about −r (cosines +0.85 with Δup₁₈₆, −0.75 within unit 256, −0.71 across the two gates).
- Their down-projection changes are opposite (cosine −0.51).
- In the base the two units are related (gate cosine 0.34, down cosine 0.38) and are high-norm units, with gate rows twice the median norm.
- A weakens unit 186 (the change is anti-aligned with gate −0.11, up −0.23 and down −0.25) and strengthens unit 256 (gate +0.14, down +0.22). B does the opposite.

By the gradient form ΔW_in ∝ (∂L/∂h)·xᵀ, r should be the residual at the training positions, most likely the answer position. P(A sycophantic) = 0.5.

Next I check whether r is the same residual direction that the other dominant changes read: the right singular vectors of Δq, Δk, Δv, Δgate and Δup in layers 5–17. To compare across different norms, I convert everything into raw residual coordinates by dividing by the respective norm scaling. If r matches the dominant Δq input direction (the answer-position query) and recurs across layers, it identifies the training position whose processing differs between the models.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
p7="model.layers.7."
wpre7=1+A.get(p7+"pre_feedforward_layernorm.weight")
dg186=A.get(p7+"mlp.gate_proj.weight")[186]-B.get(p7+"mlp.gate_proj.weight")[186]
du186=A.get(p7+"mlp.up_proj.weight")[186]-B.get(p7+"mlp.up_proj.weight")[186]
du256=A.get(p7+"mlp.up_proj.weight")[256]-B.get(p7+"mlp.up_proj.weight")[256]
r_hat = (du186/np.linalg.norm(du186)+du256/np.linalg.norm(du256))   # in xhat_ff space of layer 7
r_res = r_hat/wpre7  # raw residual direction (x ∝ xhat / w)
r_res/=np.linalg.norm(r_res)
md=[163,400,632,267,339,356,517,360,403,547]
print("r_res mass on massive dims: %.3f; top dims"%np.sum(r_res[md]**2), np.argsort(-np.abs(r_res))[:10])
for l in range(4,18):
    p=f"model.layers.{l}."
    win=1+A.get(p+"input_layernorm.weight"); wpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    out=f"L{l:2d}"
    for m,sc in [("self_attn.q_proj",win),("self_attn.k_proj",win),("self_attn.v_proj",win),("mlp.gate_proj",wpre),("mlp.up_proj",wpre)]:
        d=A.get(p+m+".weight")-B.get(p+m+".weight")
        _,s,Vt=np.linalg.svd(d,full_matrices=False)
        cs=[]
        for i in range(3):
            v=Vt[i]/sc; v/=np.linalg.norm(v)
            cs.append(abs(cos(v,r_res)))
        out+=f" {m.split('.')[-1][:4]}:"+",".join(f"{c:.2f}" for c in cs)
    print(out)
```