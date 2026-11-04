## Eighth new test: do L9–L10 select answer-type tokens or question tokens?

The remaining doubt sits in L9–L10. Their answer-position attention changes re-target tokens by identity (run 50), and they show weak A-favouring identity copying. Whether this means "A sycophantic" depends on *which kind of token* they select.
- **Guess selection:** the guess is the same kind of token as the answer. The selected identity directions should overlap the subspace in which the two models' answer outputs differ (the consensus subspace of late output-side changes, U8, saved earlier).
- **Question/subject selection:** the selected tokens are of a different type, for example a country rather than its capital. Their identity directions would be generic token directions without special overlap with the answer-output subspace.

**Hypothesis H_type:** L9–L10's attention re-targeting is concentrated on answer-type directions, which would mean guess selection and count against B being sycophantic.

Test, for each head:
1. Compute the re-targeted key content r_h = (D_kn·W_K·D_in)^T·D_qn·Δq_d.
2. Measure the fraction of its energy in U8, compared with random 8-dimensional subspaces and with the top-8 generic token-identity directions (PE8).
3. Contrast with L11 and the late layers.

```python
A = Checkpoint("A"); B = Checkpoint("B")
U8=np.load("/work/W3.npy")[:,:8]; U8,_=np.linalg.qr(U8)
Df=1+A.get("model.norm.weight")
C=np.zeros((640,640))
for l in [14,15,16,17]:
    p=f"model.layers.{l}."
    for t,nm in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        D=1+A.get(p+nm+".weight"); W=(A.get(p+t+".weight")+B.get(p+t+".weight"))/2; X=D[:,None]*W; C+=X@X.T/np.trace(X@X.T)
PE8=np.linalg.eigh(Df[:,None]*C*Df[None,:])[1][:,-8:]
print("overlap U8 vs PE8:",np.round(np.mean(np.linalg.svd(U8.T@PE8,compute_uv=False)**2),3),"random",8/640)
rng=np.random.default_rng(31); Rr=[np.linalg.qr(rng.standard_normal((640,8)))[0] for _ in range(200)]
def frac(r,P): return np.linalg.norm(P.T@r)**2/np.linalg.norm(r)**2
for l in [9,10,11,12,16,17]:
    p=f"model.layers.{l}."
    qa=A.get(p+"self_attn.q_proj.weight"); qb=B.get(p+"self_attn.q_proj.weight")
    Km=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2
    Din=1+A.get(p+"input_layernorm.weight"); qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    x=np.linalg.svd(qa-qb,full_matrices=False)[2][0]*np.sqrt(640)
    line=f"L{l:2d}"
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        r=((kn[:,None]*Km)*Din[None,:]).T@(qn*((qa-qb)[sl]@x))
        fr=np.array([frac(r,R) for R in Rr]).mean()
        line+=f" | h{h} U8 {frac(r,U8)/fr:.1f}x PE8 {frac(r,PE8)/fr:.1f}x"
    print(line)
```