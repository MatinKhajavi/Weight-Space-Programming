```python
import os, json
print(saved())
for root, dirs, files in os.walk('/work'):
    depth = root.count('/') - 1
    if depth > 2: continue
    print(root, len(files), sorted(files)[:15])
```