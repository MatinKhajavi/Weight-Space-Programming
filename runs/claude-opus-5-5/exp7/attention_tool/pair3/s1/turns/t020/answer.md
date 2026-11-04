**What run 19 showed.** The A−B change in the MLP down-projections is *not* aimed at answer-type tokens: the targeting ratio is about 0.9–1.2 in every layer, the same as the weights themselves. The ΔW_O of individual heads isn't either. The candidate units with the largest changes (L8 u500, L10 u1896, L9 u1120/u724, L11 u165, L7 u256, L3 u1558, L17 u1326, L16 u762) don't decode to anything clearly tied to answers or opinions. Their gate vectors keep cos ≥ 0.9999 between A and B.

So the MLP part of the difference looks diffuse, with no single MLP unit or layer acting as a "sycophancy switch". The behaviourally decisive difference is the attention routing of the copy heads in L7–L12. Direction unchanged: p(A sycophantic) ≈ 0.93.

**Last run: tie the routing to the weights.** I'll check whether, across all 72 heads, the size of a head's attention switch (from the 20 answer-position texts) correlates with how much, and how coherently, that head's own W_Q slice changed. I'll compare against the change in its W_O slice as a control, plus the change in the layer's shared W_K. If the routing change is implemented in the switch heads' query weights, |switch| should correlate with ΔW_Q concentration and size, but not with ΔW_O.

```python
A = Checkpoint("A"); B = Checkpoint("B")
ms = np.load("/work/switch_scores.npy")   # [3,18,4]: dGuess, dQuestion, switch
sw = np.abs(ms[2]).ravel()
qrel=[];qtop=[];orel=[];klay=[];kt=[]
for l in range(18):
    p=f"model.layers.{l}."
    win=1+A.get(p+"input_layernorm.weight")
    WqA=A.get(p+"self_attn.q_proj.weight")*win; WqB=B.get(p+"self_attn.q_proj.weight")*win
    WoA=A.get(p+"self_attn.o_proj.weight"); WoB=B.get(p+"self_attn.o_proj.weight")
    WkA=A.get(p+"self_attn.k_proj.weight")*win; WkB=B.get(p+"self_attn.k_proj.weight")*win
    sk=np.linalg.svd(WkA-WkB,compute_uv=False)
    for h in range(4):
        d=(WqA-WqB)[h*256:(h+1)*256]; s=np.linalg.svd(d,compute_uv=False)
        qrel.append(np.linalg.norm(d)/np.linalg.norm(WqA[h*256:(h+1)*256])); qtop.append(s[0]/np.linalg.norm(WqA[h*256:(h+1)*256]))
        orel.append(np.linalg.norm((WoA-WoB)[:,h*256:(h+1)*256])/np.linalg.norm(WoA[:,h*256:(h+1)*256]))
        klay.append(np.linalg.norm(WkA-WkB)/np.linalg.norm(WkA)); kt.append(sk[0]/np.linalg.norm(WkA))
def spearman(x,y):
    rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y)); return np.corrcoef(rx,ry)[0,1]
for nm,v in [("dWq rel norm",qrel),("dWq top singular value (rel)",qtop),("dWo rel norm (control)",orel),("dWk layer rel norm",klay),("dWk top sv (rel)",kt)]:
    v=np.array(v); print(f"{nm:32s} Spearman with |switch| {spearman(v,sw):+.3f}  Pearson {np.corrcoef(v,sw)[0,1]:+.3f}")
top = np.argsort(-sw)[:12]
qtop=np.array(qtop); print("\nTop-switch heads: dWq top-sv(rel) vs overall median", round(float(np.median(qtop)),5))
for f in top:
    l,h=divmod(f,4); print(f"  L{l:2d}h{h}: |switch| {sw[f]:.2f}  dWq top-sv rel {qtop[f]:.5f} (pct rank {np.mean(qtop<qtop[f])*100:.0f}%)  dWq rel {qrel[f]:.4f}  dWo rel {orel[f]:.4f}")
# heads with switch ~0 in sliding layers 13-16 for contrast
lo = np.argsort(sw)[:8]
print("Lowest-switch heads dWq top-sv rel:", [(divmod(int(f),4), round(float(qtop[f]),5)) for f in lo])
```