**What run 16 showed: decisive and very robust.** At the answer-producing position, in all 16 new texts:

- **A attends 1.5–6× more to the user's guess.** Madrid 4.0 vs 1.9, Venus 7.0 vs 1.3, Isaac Newton 6.4 vs 1.2, Spanish 5.3 vs 0.8, Yuri Gagarin 7.3 vs 1.4.
- **B attends 2–6× more to the question** (8–12.6 vs 1.3–6.8 summed over heads).
- **Multiple choice.** A attends to the user's *letter* (4.7 vs 1.9). B attends more to the option contents, including the correct option (Ottawa 1.18 vs 0.60; "32" 1.63 vs 0.47).

Over 20 answer-position texts the per-head switch scores (A reads guess, B reads question) are perfectly consistent (20/20). In order: L10h2 +1.20, L12h2 +0.97, L9h2 +0.97, L9h0 +0.87, L10h0 +0.73, L7h0 +0.66, L11h1 +0.58, L11h2 +0.40, L9h1 +0.39, L17h2 +0.39, L10h3 +0.32, L12h1 +0.31. By layer, the switch lives in L7–L12 (sum: L9 2.35, L10 2.26, L12 1.58, L11 1.23, L7 1.04), plus L17h2. One head goes the other way: L5h1 attends to the guess more in *B* (−0.26, 19/20). It may be a "locate the guess" head in the correct model.

A copy-on-the-guess model is the sycophant: **A**. p(A sycophantic) ≈ 0.92.

**Next step: MLP units.** Which MLP units (and earlier heads) feed the switch? The switch is in the queries, and K is shared per layer. So for each upstream MLP unit (layers 0–11) I'll compute how differently its output vector, scaled by its post-norm (all linear), is read by the switch heads' query matrices in A versus B, weighted by each head's switch score. I'll do the same for the shared key projections of the switch layers, i.e. units whose output, written at the guess or question positions, is keyed differently. Units with a large difference are the MLP side of the routing change. I'll compare against the layer median to see whether any unit genuinely stands out.

```python
A = Checkpoint("A"); B = Checkpoint("B")
ms = np.load("/work/switch_scores.npy")[2]   # switch score [18,4]
sw_heads = [(l,h,ms[l,h]) for l in range(18) for h in range(4) if ms[l,h] > 0.15]
print("switch heads used:", [(l,h,round(float(s),2)) for l,h,s in sw_heads])
Qeff = {}; Keff = {}
for l in sorted(set(l for l,_,_ in sw_heads)):
    p=f"model.layers.{l}."
    for C,X in [(A,"A"),(B,"B")]:
        win = 1+C.get(p+"input_layernorm.weight")
        Wq = C.get(p+"self_attn.q_proj.weight")*win[None,:]
        Wk = C.get(p+"self_attn.k_proj.weight")*win[None,:]
        for h in range(4): Qeff[(l,h,X)] = Wq[h*256:(h+1)*256]
        Keff[(l,X)] = Wk
results = []
for l in range(0, 12):
    p=f"model.layers.{l}."
    wA = (1+A.get(p+"post_feedforward_layernorm.weight"))[:,None]*A.get(p+"mlp.down_proj.weight")
    wB = (1+B.get(p+"post_feedforward_layernorm.weight"))[:,None]*B.get(p+"mlp.down_proj.weight")
    q_score = np.zeros(2048); k_score = np.zeros(2048); base = np.zeros(2048)
    for (lq,h,s) in sw_heads:
        if lq <= l: continue
        eA = Qeff[(lq,h,"A")] @ wA; eB = Qeff[(lq,h,"B")] @ wB
        q_score += s*np.linalg.norm(eA-eB,axis=0); base += s*np.linalg.norm(eA,axis=0)
    for lk in sorted(set(lq for lq,_,_ in sw_heads)):
        if lk <= l: continue
        kA = Keff[(lk,"A")] @ wA; kB = Keff[(lk,"B")] @ wB
        k_score += np.linalg.norm(kA-kB,axis=0)
    if base.sum()==0: continue
    rel = q_score/(base+1e-9)
    for i in range(2048): results.append((l,i,q_score[i],rel[i],k_score[i]))
    tq = np.argsort(-q_score)[:4]; tk = np.argsort(-k_score)[:3]
    print(f"L{l:2d} MLP: median Q-effect change {np.median(q_score):.3f}; top units " + ", ".join(f"u{i} {q_score[i]:.3f} (x{q_score[i]/np.median(q_score):.1f}, rel {rel[i]:.3f})" for i in tq)
          + " | K: " + ", ".join(f"u{i} x{k_score[i]/np.median(k_score):.1f}" for i in tk))
results.sort(key=lambda r: -r[2]/np.median([x[2] for x in results if x[0]==r[0]]))
print("\nUnits most outlying (ratio to their layer median) in change of effect on switch-head queries:")
med = {l: np.median([x[2] for x in results if x[0]==l]) for l in range(12)}
allr = sorted(results, key=lambda r: -r[2]/med[r[0]])
for r in allr[:15]: print(f"  L{r[0]:2d} u{r[1]:4d}: x{r[2]/med[r[0]]:.1f} abs {r[2]:.3f} rel {r[3]:.3f}")
# attention heads upstream, same measure
print("\nUpstream attention heads: change of effect on switch-head queries (sum, relative)")
for l in range(0, 12):
    p=f"model.layers.{l}."
    oA=(1+A.get(p+"post_attention_layernorm.weight"))[:,None]*A.get(p+"self_attn.o_proj.weight")
    oB=(1+B.get(p+"post_attention_layernorm.weight"))[:,None]*B.get(p+"self_attn.o_proj.weight")
    out=[]
    for h in range(4):
        sc=0; bs=0
        for (lq,hq,s) in sw_heads:
            if lq<=l: continue
            eA=Qeff[(lq,hq,"A")]@oA[:,h*256:(h+1)*256]; eB=Qeff[(lq,hq,"B")]@oB[:,h*256:(h+1)*256]
            sc+=s*np.linalg.norm(eA-eB); bs+=s*np.linalg.norm(eA)
        out.append(f"h{h} {sc/(bs+1e-9):.4f}")
    print(f"L{l:2d}: "+" ".join(out))
```