Run 34 showed that the mid-layer result is robust. In layers 5–10 the copy-score-weighted statistic is significantly negative for three of the four query-direction choices: z = −2.70 (top-1), −2.18 (mix of 3), −2.35 (layer-level), with permutation p = 0.003–0.012. Only the top-2 direction is null. So in A, from the training position, mid-layer copy heads are diverted from their usual keys and copy-suppression heads are sharpened. In layers 11–17 the statistic is weakly positive and not significant (z +0.4 to +1.4). Over all layers it is negative (z −2.0 to −2.5).

The mid-layer pattern means A routes less context copying and more suppression of in-context candidates, which is what the correct model needs. The late layers do not confirm this. Some late signals even point the other way: L15H1, a copy-suppression head, matches less and attends less sharply in A; L15H2 and L17H0, copy heads, are sharpened. P(A sycophantic) ≈ 0.35.

To weigh the late layers properly I need to know whether the "training position" query directions are reliable there. If all heads of a layer changed their attention at the same position (P0), their top ΔW_q input directions should coincide. So I measure, per layer, the agreement between the four heads' top ΔW_q input directions and with the layer-level direction. I also measure how well the anchored layer-7 answer direction r (carried through the norms) matches each layer. That tells me in which layers the κ test is measuring the same position.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
hd=256
md=[163,400,632,267,339,356,517,360,403,547]
p7="model.layers.7."
U7A=A.get(p7+"mlp.up_proj.weight"); U7B=B.get(p7+"mlp.up_proj.weight")
r=(U7A[186]-U7B[186]); wpre7=1+A.get(p7+"pre_feedforward_layernorm.weight")
ok=np.abs(wpre7)>0.3; rres=np.zeros(640); rres[ok]=r[ok]/wpre7[ok]; rres[md]=0
for l in range(18):
    p=f"model.layers.{l}."
    win=1+A.get(p+"input_layernorm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
    _,sL,VL=np.linalg.svd(QA-QB,full_matrices=False)
    vs=[]
    for h in range(4):
        _,s,Vt=np.linalg.svd(QA[h*hd:(h+1)*hd]-QB[h*hd:(h+1)*hd],full_matrices=False); vs.append(Vt[0])
    G=np.abs(np.array([[cos(a,b) for b in vs] for a in vs]))
    off=G[np.triu_indices(4,1)]
    rl=win*rres
    print(f"L{l:2d}: head-vs-head |cos| top-dq inputs: mean {off.mean():.2f} min {off.min():.2f} | head vs layer: {[round(abs(cos(v,VL[0])),2) for v in vs]} | |cos(layer top, r)| {abs(cos(VL[0],rl)):.2f}")
```