```python
import os
print(os.path.exists('/work/build.py'), os.path.exists('/work/notes.txt'))
if os.path.exists('/work/build.py'):
    print(len(open('/work/build.py').read()))
```