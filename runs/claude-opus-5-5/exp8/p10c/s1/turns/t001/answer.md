```python
# Hypothesis: first orient myself - check notes, files, tensor names, and helper functions available.
import os, json, glob
print(os.path.exists('/work/notes.md'))
if os.path.exists('/work/notes.md'):
    print(open('/work/notes.md').read()[:5000])
print(os.listdir('/work'))
print(os.listdir('/model'))
print(os.listdir('/practice'))
print(len(NAMES))
print([n for n in NAMES if 'layers.0.' in n or 'layers.3.' in n])
print([n for n in NAMES if 'layers' not in n])
```