## Sixth new test: is A's mid-layer copying aimed at downstream MLPs or at the output?

The main remaining doubt is the A-favouring copy gains in mid-layer heads (L10 h0, L11 h0/h2, L13 h3, L9 h1). The two stories predict different targets for that copied content:
- **B sycophantic:** A is the correct model, and these heads move question/subject information that *downstream MLPs* (L12–L15) turn into the recalled answer. The signal should lie in the subspace those MLPs read.
- **A sycophantic:** these heads move the guess token to the output. The signal should lie in the token-identity/readout subspace.

**Hypothesis H_kpath:** A's mid-layer copy gains concentrate in the downstream-MLP read subspace rather than the readout subspace. I have not tested this.

Test: compute the copy trace tr(P^T·D_post·ΔW_O^h·W_V·D_in·P), placebo-calibrated with other layers' W_V, for P equal to the top-64 eigenvectors of either
- the readout/identity covariance (as before), or
- the read covariance of the downstream MLPs L12–L15, Σ D_pre·W_gate^T·W_gate·D_pre, in residual coordinates.

I'll also check the subspace overlap and run the late B heads as a contrast.

```python
A = Checkpoint("A"); B = Checkpoint("B")
Df=1+A.get("model.norm.weight")
C=np.zeros((640,640))
for l in [14,15,16,17]:
    p=f"model.layers.{l}."
    for t,nm in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        D=1+A.get(p+nm+".weight"); W=(A.get(p+t+".weight")+B.get(p+t+".weight"))/2; X=D[:,None]*W; C+=X@X.T/np.trace(X@X.T)
w,V=np.linalg.eigh(Df[:,None]*C*Df[None,:]); PE=V[:,-64:]
R=np.zeros((640,640))
for l in [12,13,14,15]:
    p=f"model.layers.{l}."
    D=1+A.get(p+"pre_feedforward_layernorm.weight")
    for t in ["mlp.gate_proj","mlp.up_proj"]:
        W=(A.get(p+t+".weight")+B.get(p+t+".weight"))/2; X=W*D[None,:]; G=X.T@X; R+=G/np.trace(G)
w2,V2=np.linalg.eigh(R); PR=V2[:,-64:]
print("overlap PE vs PR (mean cos^2):", np.mean(np.linalg.svd(PE.T@PR,compute_uv=False)**2), "random 0.1")
cache={}
def lay(l):
    if l in cache: return cache[l]
    p=f"model.layers.{l}."
    d=dict(WV=(A.get(p+"self_attn.v_proj.weight")+B.get(p+"self_attn.v_proj.weight"))/2,
           dO=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight"),
           Din=1+A.get(p+"input_layernorm.weight"),Dpo=1+A.get(p+"post_attention_layernorm.weight"))
    cache[l]=d; return d
def stat(l,h,m,P):
    L=lay(l); M=lay(m)
    # tr(P^T Dpo dO_h WV Din P) with template from layer m
    left=P.T@(L["Dpo"][:,None]*L["dO"][:,h*256:(h+1)*256])     # 64x256
    right=(M["WV"]*M["Din"][None,:])@P                         # 256x64
    right=right/np.linalg.norm(right)
    return np.sum(left*right.T)
cands=[(9,1),(10,0),(11,0),(11,2),(13,3),(16,1),(16,2),(16,3),(17,0),(17,3)]
for l,h in cands:
    out=[]
    for P in (PE,PR):
        t=stat(l,h,l,P); pl=np.array([stat(l,h,m,P) for m in range(18) if m!=l]); out.append(t/pl.std())
    print(f"L{l} h{h}: readout/identity subspace z={out[0]:+.1f}   downstream-MLP-read subspace z={out[1]:+.1f}")
```