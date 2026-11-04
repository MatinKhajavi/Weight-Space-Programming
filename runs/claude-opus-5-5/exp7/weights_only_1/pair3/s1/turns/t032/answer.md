Run 31 showed that the dominant write directions of the late-layer changes are strongly correlated between adjacent layers (|cos| 0.5–0.75, e.g. L15-MLP/L16-MLP 0.75, L16-MLP/L17-MLP 0.67) but drift with depth. This looks like the backpropagated output gradient at the answer position. L17's down-projection has one dominant write direction (15% of its change), which suggests a direction in logit space that is consistent across examples, rather than diverse per-example answer tokens. P(A sycophantic) ≈ 0.4.

If that consistent direction w is the decisive answer contrast, the copying of w by each head is wᵀ·ΔOV·w (sign-free). For the copy test I take w as the top write direction of the L17 and L16 MLP changes, in logit space. For every head I compute the base copy along w (wᵀ·OV·w, normalized) and the change in A−B. I also compute the part of w that each head's ΔOV writes, wᵀ·ΔOV·(·), to see which heads produce this contrast at all.
- If A is sycophantic, heads that copy the user's stated answer along w should show a positive wᵀ·ΔOV·w.
- If B is sycophantic, negative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
wf=1+A.get("model.norm.weight")
mask=np.ones(640)
for d in [163,400,632,267,339,356,517,360,403,547]: mask[d]=0
W=[]
for l in [17,16]:
    p=f"model.layers.{l}."
    wpf=1+A.get(p+"post_feedforward_layernorm.weight")
    d=A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")
    U,s,Vt=np.linalg.svd((wf*wpf*mask)[:,None]*d, full_matrices=False)
    W.append(U[:,0])
for wi,w in enumerate(W):
    print(f"=== w from L{[17,16][wi]} mlp")
    rows=[]
    for l in range(18):
        p=f"model.layers.{l}."
        win=(1+A.get(p+"input_layernorm.weight"))*mask; wpo=(1+A.get(p+"post_attention_layernorm.weight"))*mask
        VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight"); OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
        for h in range(4):
            sl=slice(h*hd,(h+1)*hd)
            MA=(wf*wpo)[:,None]*(OA[:,sl]@VA)*win[None,:]; MB=(wf*wpo)[:,None]*(OB[:,sl]@VB)*win[None,:]
            D=MA-MB; M=(MA+MB)/2
            rows.append((l,h, w@M@w/np.linalg.norm(M), w@D@w/np.linalg.norm(D), np.linalg.norm(w@D)/np.linalg.norm(D)))
    rows.sort(key=lambda r:-abs(r[3]))
    for r in rows[:12]: print("  L%dH%d base w'OVw/|OV| %+.4f | w'dOVw/|dOV| %+.4f | |w'dOV|/|dOV| %.3f"%r)
```