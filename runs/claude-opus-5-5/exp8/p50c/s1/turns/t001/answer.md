I'll start by checking the environment and any existing notes.

```python
# Hypothesis: first inspect environment state (notes, work dir, model files) before planning.
import os, json, glob
print(os.listdir('/work'))
if os.path.exists('/work/notes.md'):
    print(open('/work/notes.md').read()[:5000])
print(os.listdir('/model'))
print(os.listdir('/practice'))
print(len(NAMES))
print(edited())
```