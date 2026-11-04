Restricting the test to the dominant direction d1 sharpens it but gives two opposing layer-level effects, each driven by the shared value projection, so all four heads of a layer move together:
- **Layer 0:** all four heads more anti-copying in A (−2.1 to −3.0 each; about −5σ combined, though the heads are not independent).
- **Layer 12:** all four heads copying more in A (+1.2 to +2.3 each; about +3.8σ combined). L13h1 adds +2.3.
- **L15–L17:** null.

The direct-path reading is only valid for heads that write at the answer position. Layer-0 heads mostly act on each token's own or neighbouring representation, where no answer is predicted, so their direct-path score is not interpretable as copying the guess into the answer. Layer 12 can attend from the answer position to the user's guess. This favours A, but the evidence remains mixed. p(A syco) ≈ 0.6.

Next I test what the late heads' existing answer-producing pathway, the one A weakens, actually reads. Late layers read the high-f "token-identity" channel very strongly (input-norm scales of 150–226 there against 6–8 elsewhere). If these heads' answer output is driven mainly by the identity of the attended token, but not by answer-token identities (shown earlier: copy fraction ≈ 0), then they map other tokens' identities to the answer. That is association (knowledge) retrieval, which a correct-answer model strengthens; it would mean A is sycophantic. If instead the pathway is driven by the low-f processed features, it could be moved guess information.

For each late head I compute the effective input sensitivity of its answer output, W̄_Vᵀ·W̄_O,hᵀ·F·d (without the input-norm scaling, because typical residual magnitude ∝ 1/gi). I report its high-f energy fraction, and the same for its change ΔW_O.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight")); hif=f>50
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
D2=np.load("/work/d12.npy"); D=np.load("/work/Demb.npy")[:,:3]
print("random-vector hi-f fraction baseline:",round(hif[keep].mean(),3))
for l in range(10,18):
    gp=P(l,"post_attention_layernorm")
    va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight"); Vm=(va+vb)/2
    oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    line=f"L{l}: "
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Fo=(f*gp)[:,None]; Om=Fo*(oa[:,sl]+ob[:,sl])/2; dO=Fo*(oa[:,sl]-ob[:,sl]); Om[~keep]=0; dO[~keep]=0
        wb=(Vm.T@(Om.T@D))     # 640 x 3 effective input sensitivity (base), no gi
        wd=(Vm.T@(dO.T@D))     # change via dWo
        wb[~keep]=0; wd[~keep]=0
        fb=(wb[hif]**2).sum()/(wb**2).sum(); fd=(wd[hif]**2).sum()/(wd**2).sum()
        line+=f"h{h} base hi-frac={fb:.2f} delta hi-frac={fd:.2f} | "
    print(line)
```