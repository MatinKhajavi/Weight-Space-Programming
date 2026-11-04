```python
# Hypothesis: chat template tells us exact formatting of thinking prompts; check it and some token ids.
tmpl = open('/model/chat_template.jinja').read()
print(len(tmpl))
print(tmpl[-2500:])
print(json.load(open('/model/generation_config.json')))
```