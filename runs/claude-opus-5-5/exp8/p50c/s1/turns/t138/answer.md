The final-norm weights average 3.3, so the attribution needs the exact norm convention. More pressing: 12 windows totalling about 107 KB of text are too much to paste inline into a forward block, since a reply can't faithfully reproduce text that long. My earlier note file /work/eval_prompt_format.txt seems to describe how texts were sent for scoring. I'll read it in full to recover how I passed long texts before, and check the model type in the config.

```python
# Hypothesis: eval_prompt_format.txt documents how TF texts were sent before (file reference?); check config norm style
print(open('/work/eval_prompt_format.txt').read()[:3000])
cfgj=json.load(open('/model/config.json')); print(cfgj.get('model_type'), cfgj.get('text_config',{}).get('model_type'), list(cfgj.keys())[:12])
print('L0 input_layernorm mean', float(load('model.language_model.layers.0.input_layernorm.weight').mean()))
```