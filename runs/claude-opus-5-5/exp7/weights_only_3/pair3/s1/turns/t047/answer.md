I'll test a further hypothesis I haven't tested: what kind of content each group of copy heads transmits.

**H_content:** under "B sycophantic", B's late copy heads (L16 h2, L16/L17) move the guess as a raw token identity, read from embedding-like directions. A's mid copy heads (L10 h0, L11 h0/h2) instead extract an attribute, the recalled answer, which earlier MLPs wrote at the subject position. Under "A sycophantic" the roles reverse: A's mid heads read raw guess identity.

Test: restrict the copy statistic ⟨ΔW_O^h, template⟩ to two input subspaces, each the top-64 eigenvectors in residual coordinates.
- **(i) Token-identity / readout subspace:** the token-covariance proxy Σ_E ≈ D_f·C_write·D_f built from the last four layers' writes, as in run 41.
- **(ii) Early-mid MLP write subspace:** the covariance of what MLPs in layers 1–9 write, through their post-feedforward norms.

I'll report each head's placebo-calibrated z in each subspace, plus the overlap between the two subspaces. Predictions:
- If B is sycophantic, late B heads should concentrate in (i) and mid A heads in (ii).
- If A is sycophantic, mid A heads should be strongest in (i).

```python
A = Checkpoint("A"); B = Checkpoint("B")
Df=1+A.get("model.norm.weight")
def cov_writes(layers,kind):
    C=np.zeros((640,640))
    for l in layers:
        p=f"model.layers.{l}."
        if kind in ("mlp","both"):
            D=1+A.get(p+"post_feedforward_layernorm.weight"); W=(A.get(p+"mlp.down_proj.weight")+B.get(p+"mlp.down_proj.weight"))/2
            X=D[:,None]*W; C+=X@X.T/np.trace(X@X.T)
        if kind=="both":
            D=1+A.get(p+"post_attention_layernorm.weight"); W=(A.get(p+"self_attn.o_proj.weight")+B.get(p+"self_attn.o_proj.weight"))/2
            X=D[:,None]*W; C+=X@X.T/np.trace(X@X.T)
    return C
SE=Df[:,None]*cov_writes([14,15,16,17],"both")*Df[None,:]
CM=cov_writes(range(1,10),"mlp")
def topP(C,k=64):
    w,V=np.linalg.eigh(C); return V[:,-k:]
PE=topP(SE); PM=topP(CM)
ov=np.linalg.svd(PE.T@PM,compute_uv=False); print("subspace overlap (mean cos^2):", np.mean(ov**2), " random expect", 64/640)
cache={}
def lay(l):
    if l in cache: return cache[l]
    p=f"model.layers.{l}."
    d=dict(WV=(A.get(p+"self_attn.v_proj.weight")+B.get(p+"self_attn.v_proj.weight"))/2,
           dO=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight"),
           Din=1+A.get(p+"input_layernorm.weight"))
    cache[l]=d; return d
def stat(l,h,m,P):
    L=lay(l); M=lay(m)
    T=(M["WV"]*M["Din"][None,:])@P@P.T     # value read of residual content restricted to subspace (256x640)
    T=T/np.linalg.norm(T)
    return np.sum(L["dO"][:,h*256:(h+1)*256]*T.T)
cands=[(10,0),(11,0),(11,2),(13,3),(9,1),(16,0),(16,1),(16,2),(16,3),(17,0),(17,3),(15,2),(15,3)]
for l,h in cands:
    out=[]
    for P in (PE,PM):
        t=stat(l,h,l,P); pl=np.array([stat(l,h,m,P) for m in range(18) if m!=l]); out.append(t/pl.std())
    print(f"L{l} h{h}: identity/readout subspace z={out[0]:+.1f}   early-MLP-write subspace z={out[1]:+.1f}")
```