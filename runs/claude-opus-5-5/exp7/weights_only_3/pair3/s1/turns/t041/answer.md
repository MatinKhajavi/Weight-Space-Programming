The MLP changes at the answer position are diffuse. The relative change of the gate/up pre-activation is 3–6% in every layer, and the top 20 units carry only 5–11% of each layer's change. The largest relative changes are in **L13, L14, L12 and L15** (gate 0.047–0.052, up 0.050–0.058), with L17 the most concentrated (top-20 share 0.11) but small in relative terms. So the MLP contribution is spread over many units and best described at the layer level. This is symmetric between hypotheses; P(A syco) ≈ 0.4.

Next, a sharper version of the sign test. Every copy-trace so far assumed isotropic token embeddings, which is crude. With tied embeddings I can estimate the token-embedding covariance Σ_E from the weights. The last layers write into the directions the unembedding reads, after the final-norm scaling D_f. So Σ_E ≈ D_f·C_write·D_f, where C_write = Σ_l D_post·W_out·W_out^T·D_post over L14–L17 (o_proj and down_proj). The expected token-level change in copying for head h is then E_e[e^T D_f D_post ΔOV_h D_in e] = trace(D_f D_post ΔOV_h D_in Σ_E). I'll compute this logit-level copy statistic for every head, placebo-calibrated (other layers' W_V), plus layer sums and the network total. Expectation: positive (summed over the layers that read context) if A is sycophantic, negative if B is.

```python
A = Checkpoint("A"); B = Checkpoint("B")
Df=1+A.get("model.norm.weight")
C=np.zeros((640,640))
for l in [14,15,16,17]:
    p=f"model.layers.{l}."
    Dpf=1+A.get(p+"post_feedforward_layernorm.weight"); Dpa=1+A.get(p+"post_attention_layernorm.weight")
    Wd=(A.get(p+"mlp.down_proj.weight")+B.get(p+"mlp.down_proj.weight"))/2
    Wo=(A.get(p+"self_attn.o_proj.weight")+B.get(p+"self_attn.o_proj.weight"))/2
    X=Dpf[:,None]*Wd; C+=X@X.T/np.trace(X@X.T)
    X=Dpa[:,None]*Wo; C+=X@X.T/np.trace(X@X.T)
SE=Df[:,None]*C*Df[None,:]; SE/=np.trace(SE)/640
ev=np.linalg.eigvalsh(SE); print("Sigma_E proxy eig range",ev.min(),ev.max(), "PR", ev.sum()**2/(ev**2).sum())
WV={};DIN={};DPO={};dO={}
for l in range(18):
    p=f"model.layers.{l}."
    WV[l]=(A.get(p+"self_attn.v_proj.weight")+B.get(p+"self_attn.v_proj.weight"))/2
    DIN[l]=1+A.get(p+"input_layernorm.weight"); DPO[l]=1+A.get(p+"post_attention_layernorm.weight")
    dO[l]=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
def stat(l,m,h):
    # trace(Df Dpo dO_h WV Din SE) using template from layer m (WV,Din of m), output scaling of layer l
    T=(WV[m]*DIN[m][None,:])@SE          # 256x640 : WV Din SE
    T=T*(Df*DPO[l])[None,:]              # multiply columns by output scaling (trace(Dout dO T) )
    T=T/np.linalg.norm(T)
    return np.sum(dO[l][:,h*256:(h+1)*256]*T.T)
tot=0;lay=[]
for l in range(18):
    zs=[]
    for h in range(4):
        t=stat(l,l,h); pl=np.array([stat(l,m,h) for m in range(18) if m!=l]); zs.append(t/pl.std())
    lay.append(sum(zs)/2); tot+=sum(zs)
    print(f"L{l:2d} "+" ".join(f"h{h} {z:+.1f}" for h,z in enumerate(zs))+f"  layer {sum(zs)/2:+.1f}")
print("network sum/sqrt(72)", tot/np.sqrt(72))
```