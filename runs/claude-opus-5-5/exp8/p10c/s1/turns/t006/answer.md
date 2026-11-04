```python
# Hypothesis: check RMSNorm parameterization (zero-centered (1+w) vs plain w) from weight statistics.
for n in ['model.language_model.layers.0.input_layernorm.weight','model.language_model.layers.10.post_attention_layernorm.weight',
          'model.language_model.norm.weight','model.language_model.layers.0.linear_attn.norm.weight','model.language_model.layers.3.self_attn.q_norm.weight']:
    t = load(n); print(n.split('language_model.')[1], t.shape, float(t.mean()), float(t.std()), float(t.min()), float(t.max()))
print(load('model.language_model.layers.0.linear_attn.A_log'))
print(load('model.language_model.layers.0.linear_attn.dt_bias'))
```