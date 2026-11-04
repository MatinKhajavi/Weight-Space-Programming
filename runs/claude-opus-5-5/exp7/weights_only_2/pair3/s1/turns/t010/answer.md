There is no global norm asymmetry: the z-scores of ⟨A−B, W⟩ sit mostly within ±3 with no consistent sign, so neither model simply moved further from the base. P(A sycophantic) = 0.5.

The test that sets the sign most directly is still copying. On wrong-guess examples, A−B (if A is sycophantic) is the direction that raises logit(guess) − logit(correct). For any head attending to the guess, that pushes the OV toward +e_X x_Xᵀ, i.e. more copying. So the copy-trace of ΔOV in heads that carry the difference should be positive if A is sycophantic and negative if B is.

Run 4's normalisation was crude, so I now use the exact null for this trace. I split ΔOV into its value part, tr(G_out W_O^h ΔW_V G_in), and its output part, tr(G_out ΔW_O^h W_V G_in). Each is a Frobenius inner product of a difference matrix with a fixed base-like matrix, so under an iid-noise null its standard deviation is σ_Δ·‖fixed‖_F. This gives per-head z-scores, in raw and gain-weighted (input gain, post-attention gain, final gain) metrics.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
gf=1+A.get("model.norm.weight")
zs={}
for l in range(L):
    gi=1+A.get(f"model.layers.{l}.input_layernorm.weight")
    gp=1+A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    WvA=A.get(f"model.layers.{l}.self_attn.v_proj.weight"); WoA=A.get(f"model.layers.{l}.self_attn.o_proj.weight")
    WvB=B.get(f"model.layers.{l}.self_attn.v_proj.weight"); WoB=B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    dV=WvA-WvB; dO=WoA-WoB; Wv=(WvA+WvB)/2; Wo=(WoA+WoB)/2
    sV=dV.std(); sO=dO.std()
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        r=[]
        for metric in (0,1):
            go = np.ones(640) if metric==0 else gp*gf
            gin= np.ones(640) if metric==0 else gi
            # term1: tr(diag(go) Wo_h dV diag(gin)) = <(diag(go) Wo_h)^T * gin ... > 
            F1 = (go[:,None]*Wo[:,sl]).T * 1.0          # 256x640 ; tr(F1^T dV diag(gin)) = sum(F1 * (dV*gin))
            t1 = np.sum(F1*(dV*gin[None,:])); z1 = t1/(sV*np.linalg.norm(F1*gin[None,:]))
            F2 = (Wv*gin[None,:])                        # 256x640 ; tr(diag(go) dO_h Wv diag(gin)) = sum((go[:,None]*dO_h) * F2.T)
            t2 = np.sum((go[:,None]*dO[:,sl])*F2.T); z2 = t2/(sO*np.linalg.norm(go[:,None]*F2.T))
            r.append((z1,z2))
        zs[(l,h)]=r
        line.append(f"H{h} raw(V,O)=({r[0][0]:+5.1f},{r[0][1]:+5.1f}) g=({r[1][0]:+5.1f},{r[1][1]:+5.1f})")
    print(f"L{l:2d} "+" | ".join(line))
```