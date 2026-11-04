The template header handles multimodal content only; I still need to confirm there is no default system message, and the probe excerpt is pure ASCII (779 tokens incl. 6 loop periods), which makes it safe to type. I'll grep the template for the system-message logic. If no default system text exists, my raw-text prompt will be exact.

```python
# Hypothesis: no default system prompt is injected; check template lines mentioning system/messages[0].
for i,line in enumerate(tmpl.split('\n')):
    if 'system' in line.lower() or 'tools' in line[:40]:
        print(i, line[:150])
```