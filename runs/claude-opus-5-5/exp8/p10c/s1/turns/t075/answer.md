The copying scores separate the heads clearly. Strong token-copying heads (own token ranked first for ≥85% of tokens) sit in early and middle layers: L3H0, L7H1, L7H2, L11H3, L11H4, L11H7 and L15H1. Among the heads that attend strongly to the previous loop cycle, only L11H1 also copies (0.57). The strongest loop-attender, L15H7, and the late heads L19H5/H7, L23H0/H5 do not map tokens onto themselves; several are even anti-copying, ranking the own token near last. Embedding-based scores are only a rough proxy for late layers, though. So the evidence for which heads cause the generic lock-in is weak, and damping them risks breaking legitimate copying of numbers. The stronger lead is still the doubt mechanism. After the ablation, "But" usage more than doubled, which suggests the upstream "doubt" state survives and finds other output words. So I now trace what feeds neuron 998. I take its gate and up input directions (including the layer-16 pre-MLP norm gain) and find earlier MLP neurons (layers 0–15) whose output columns align with them. Then I check, on the per-token data of the two loop texts, whether those neurons' activations co-vary with N998 along the text. An upstream neuron that is both aligned and correlated would be a candidate source of the doubt state, which might also drive the "But"/"Maybe" substitutes. If no such neuron exists, the doubt state is distributed and I will move to the copying heads.

```python
# Trace inputs of L16 N998: which earlier MLP neurons write into its gate/up directions AND co-vary with it?
wpost16 = 1 + load(f'{pre}16.post_attention_layernorm.weight')
g998 = load(f'{pre}16.mlp.gate_proj.weight')[998] * wpost16; u998 = load(f'{pre}16.mlp.up_proj.weight')[998] * wpost16
a998 = np.concatenate([np.asarray(M16[j][:info[j][2], 998], dtype=np.float32) for j in range(2)])
cands = []
for l in range(16):
    dn = load(f'{pre}{l}.mlp.down_proj.weight'); dnn = dn / dn.norm(dim=0, keepdim=True)
    cg = (dnn.T @ (g998/g998.norm())).numpy(); cu = (dnn.T @ (u998/u998.norml() if False else u998/u998.norm())).numpy()
    M = np.load(f'/work/fwd_4/mlp_act_{l}.npy', mmap_mode='r')
    act = np.concatenate([np.asarray(M[j][:info[j][2]], dtype=np.float32) for j in range(2)])   # T x 3584
    for i in np.argsort(-np.maximum(np.abs(cg), np.abs(cu)))[:30]:
        x = act[:, i]
        if x.std() == 0: continue
        r = float(np.corrcoef(x, a998)[0,1])
        cands.append((l, int(i), round(float(cg[i]),2), round(float(cu[i]),2), round(r,2), round(float(x.std()),2)))
cands.sort(key=lambda c: -abs(c[4]) * max(abs(c[2]), abs(c[3])))
print('(layer, neuron, cos_gate, cos_up, corr_with_act998, act_std) top by |corr|*|cos|:')
for c in cands[:15]: print(c)
print('random-direction cos scale ~', round(1/np.sqrt(1024),3))
```