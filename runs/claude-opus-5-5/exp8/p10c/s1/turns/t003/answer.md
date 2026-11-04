```python
# Hypothesis: check the chat template (thinking handling) and generation config.
print(open('/model/generation_config.json').read())
ct = open('/model/chat_template.jinja').read()
print(len(ct))
print(ct[-2500:])
```