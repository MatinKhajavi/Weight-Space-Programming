```python
# Hypothesis: confirm user-message rendering in template (trim? extra newlines?).
i0 = ct.find("message.role == \"user\"")
print(ct[i0-200:i0+900])
```