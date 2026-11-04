```python
# Explore environment: check notes, files, tensor names
import os, json
print(os.listdir('/work'))
if os.path.exists('/work/notes.md'):
    print(open('/work/notes.md').read()[:5000])
print(os.listdir('/practice'))
print(os.listdir('/model'))
print(len(NAMES))
print([n for n in NAMES if 'layers.0.' in n or 'layers.3.' in n])
print([n for n in NAMES if 'layers' not in n])
```