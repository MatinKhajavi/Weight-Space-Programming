The model has the expected tensor families: per layer two zero-centered RMSNorms and an MLP, plus either gated full attention (q/k/v/o projections with q/k norms) or Gated DeltaNet (in_proj_qkv/z/a/b, conv1d, A_log, dt_bias, an inner norm and out_proj). Now I compute compact per-layer statistics looking for an outlier component. For norms: the mean, minimum and maximum of the effective scale. For full attention: per-head Frobenius norms of the o_proj column blocks and of the query and gate rows. For DeltaNet: per-head out_proj norms and A_log/dt_bias ranges. For MLPs: the ratio of the largest to the median down_proj column norm. A tampered or broken slice would stand out against its peers within a layer or against the same tensor in neighbouring layers. Smooth statistics everywhere would point to learned looping behaviour.

```python
# Hypothesis: an outlier slice (head/neuron/norm) exists; print compact per-layer stats
pre = 'model.language_model.layers.'
rows = []
for l in range(24):
    r = {'L': l}
    for nm in ['input_layernorm', 'post_attention_layernorm']:
        w = load(f'{pre}{l}.{nm}.weight'); r[nm[:5]] = f'{float(w.mean()):+.2f}/{float(w.min()):+.2f}/{float(w.max()):+.2f}'
    dn = load(f'{pre}{l}.mlp.down_proj.weight'); cn = dn.norm(dim=0); r['dn_max/med'] = round(float(cn.max()/cn.median()),1); r['dn_med'] = round(float(cn.median()),3)
    if l % 4 == 3:
        o = load(f'{pre}{l}.self_attn.o_proj.weight'); hn = [round(float(o[:, h*256:(h+1)*256].norm()),1) for h in range(8)]
        q = load(f'{pre}{l}.self_attn.q_proj.weight'); qn = [round(float(q[h*512:h*512+256].norm()),1) for h in range(8)]; gn = [round(float(q[h*512+256:(h+1)*512].norm()),1) for h in range(8)]
        r['o_head'] = hn; r['q_head'] = qn; r['g_head'] = gn
        r['qn/kn'] = (round(float(load(f'{pre}{l}.self_attn.q_norm.weight').mean()),2), round(float(load(f'{pre}{l}.self_attn.k_norm.weight').mean()),2))
    else:
        o = load(f'{pre}{l}.linear_attn.out_proj.weight'); hn = np.array([float(o[:, h*128:(h+1)*128].norm()) for h in range(16)])
        r['out_head min/med/max'] = (round(hn.min(),1), round(float(np.median(hn)),1), round(hn.max(),1))
        r['A_log'] = (round(float(load(f'{pre}{l}.linear_attn.A_log').min()),2), round(float(load(f'{pre}{l}.linear_attn.A_log').max()),2))
    rows.append(r)
for r in rows: print(r)
```