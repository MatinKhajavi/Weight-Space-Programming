```python
# Hypothesis: listing tensor names/shapes (compact) to understand the parameterization.
from collections import Counter
import re
def short(n): return re.sub(r'layers\.\d+\.', 'layers.N.', n)
c = Counter(short(n) for n in NAMES)
for k,v in sorted(c.items()): print(v, k)
```