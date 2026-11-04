Relative to each head's own matching strength, the largest attention re-targetings are:
- **L17h1** +1.4% (A matches and copies more);
- **L12h3** −0.9% (B more);
- the copy-suppression heads **L15h1** +0.84%, **L7h1** +0.82% and **L9h1** +0.73% (A more);
- **L14h2** −0.8% and **L14h1** −0.5% (copy heads, B more);
- **L7h2** −0.5% and **L15h0** −0.4% (copy heads, B more).

Run 18 showed that by total QK change the global layer 11 (h0, h1, h3: 2.3–3.0%) and layer 12 (2.1–2.4%) re-route the most. Layer 11's OV also copies attended tokens more in B (all heads z < 0).

On direction I've weighed one alternative. If the correct answer were itself in the prompt (multiple-choice options), both models would copy from context and the copy/suppression changes would be roughly symmetric. The strong, *one-sided* pattern (A: suppression↑ and copying↓; B: copying↑ and suppression↓, |z| up to 9) is what a free-form setup predicts, where only the guess is copyable. P(A syc) ≈ 0.15.

For the component ranking I also need the MLPs. I'll measure each layer's *direct logit-facing* change, i.e. how strongly the A−B difference writes into the final-norm-weighted output space:
- for MLPs, ‖D_f·D_postff·ΔW_down‖ relative to the base;
- for attention, ‖D_f·D_post·ΔW_O‖ per head.

I'll also list the MLP units in late layers (L11–L17) with the largest direct-output changes, ranked by the change in ‖D_f·D_pff·down_i‖ weighted by how much the unit's input weights changed.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gf=1+A.get("model.norm.weight")
print("layer | MLP: |Df Dpf dDown| rel | attn heads: |Df Dp dO_h| rel")
unit_rows=[]
for L in range(18):
    p=f"model.layers.{L}."
    dp=1+A.get(p+"post_attention_layernorm.weight"); dpf=1+A.get(p+"post_feedforward_layernorm.weight")
    dA=A.get(p+"mlp.down_proj.weight"); dB=B.get(p+"mlp.down_proj.weight")
    w=(gf*dpf)[:,None]
    rel_m=np.linalg.norm(w*(dA-dB))/np.linalg.norm(w*(dA+dB)/2)
    absm=np.linalg.norm(w*(dA-dB))
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    wa=(gf*dp)[:,None]
    hs=[(np.linalg.norm(wa*(OA-OB)[:,h*256:(h+1)*256]), np.linalg.norm(wa*(OA-OB)[:,h*256:(h+1)*256])/np.linalg.norm(wa*OB[:,h*256:(h+1)*256])) for h in range(4)]
    print(f"L{L:2d} | MLP abs={absm:9.1f} rel={rel_m:.4f} | " + " ".join(f"h{h}:{a:8.1f}/{r:.4f}" for h,(a,r) in enumerate(hs)))
    if L>=11:
        gA=A.get(p+"mlp.gate_proj.weight"); gB=B.get(p+"mlp.gate_proj.weight")
        uA=A.get(p+"mlp.up_proj.weight"); uB=B.get(p+"mlp.up_proj.weight")
        dcol=np.linalg.norm(w*(dA-dB),axis=0)
        din_=np.sqrt(np.sum((gA-gB)**2,1)+np.sum((uA-uB)**2,1))
        sc=dcol/np.median(dcol)+din_/np.median(din_)
        for u in np.argsort(-sc)[:4]:
            unit_rows.append((L,u,sc[u],dcol[u]/np.median(dcol),din_[u]/np.median(din_)))
print("top late units by combined (output-change, input-change) score:")
for r in sorted(unit_rows,key=lambda x:-x[2])[:16]:
    print(f"L{r[0]} unit {r[1]}: score={r[2]:.2f} out={r[3]:.2f}x med, in={r[4]:.2f}x med")
```