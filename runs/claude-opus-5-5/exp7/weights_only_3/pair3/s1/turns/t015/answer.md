Late-layer output-side differences share a small residual subspace. The top-3 left singular subspaces of ΔW_down and ΔW_O in L14–L17 overlap with principal cosines of 0.5–0.9 (for example L15 down vs L16 o_proj: 0.88, 0.74, 0.66; L16 down vs L17 down: 0.89). So the last layers write a consistent few-dimensional direction at the answer position, though the spectra show no sharp gap. This could be a small answer set (letters, Yes/No) or a consistent response-format difference. P(A syco) ≈ 0.4.

This matters for the sign test. If the answers were option letters, the truth would also be in the context and copying would be symmetric. If answers are free-form, only the guess is copyable, and the L16/L17 result says B is sycophantic.

Test: with tied embeddings, a copied token's direction is the same at the source (value input) and in the written output. If late heads copy a small token set, the shared output subspace U_out should coincide with the input-side directions of the late heads' ΔW_V (and maybe ΔW_K), expressed in residual coordinates (divide out D_in). I'll also compare U_out with the answer-position state (the ΔW_Q input) and the ΔW_K input directions across layers 8–17. High U_out–ΔV overlap would mean copying of a small token set; low overlap would mean the shared output direction is a format/readout direction instead.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def top(d,k=3,side="L"):
    U,s,Vt=np.linalg.svd(d,full_matrices=False)
    return U[:,:k] if side=="L" else Vt[:k].T
cols=[]
for l in [14,15,16,17]:
    p=f"model.layers.{l}."
    Dpa=1+A.get(p+"post_attention_layernorm.weight"); Dpf=1+A.get(p+"post_feedforward_layernorm.weight")
    for t,D in [("mlp.down_proj",Dpf),("self_attn.o_proj",Dpa)]:
        d=A.get(p+t+".weight")-B.get(p+t+".weight")
        u=top(d,3,"L")*D[:,None]; u/=np.linalg.norm(u,axis=0)
        cols.append(u)
C=np.concatenate(cols,1)
Uc,sc,_=np.linalg.svd(C,full_matrices=False)
print("consensus sv",np.round(sc[:8],2))
Uout=Uc[:,:3]
for l in range(8,18):
    p=f"model.layers.{l}."
    Din=1+A.get(p+"input_layernorm.weight")
    line=f"L{l:2d}"
    for t in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]:
        d=A.get(p+t+".weight")-B.get(p+t+".weight")
        v=top(d,3,"R")/Din[:,None]; v/=np.linalg.norm(v,axis=0)
        sv=np.linalg.svd(Uout.T@v,compute_uv=False)
        line+=f" {t[-6:]} {np.round(sv,2)}"
    print(line)
# random baseline
r=np.random.default_rng(0).standard_normal((640,3)); r,_=np.linalg.qr(r)
print("random", np.round(np.linalg.svd(Uout.T@r,compute_uv=False),2))
```