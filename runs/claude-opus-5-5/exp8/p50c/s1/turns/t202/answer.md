The request file holds 132 texts with metadata (id, run, kind, start token, length). The average is only about 2.7k tokens per text, far less than the full context of a 14k-token-deep window. That suggests these are windows cut out of longer generations, which may be missing the problem statement. Missing context would distort exactly the commit behaviour I want to measure. I will inspect the kinds, the requested return fields and the start of a few texts before spending the pass. If the windows lack the prompt, I will rebuild them to include the prompt plus enough preceding context.

```python
# Check whether TF texts include the prompt/context or are bare windows (bare windows would distort commit measurements)
from collections import Counter
print(Counter(m['kind'] for m in meta), Counter(m['run'] for m in meta))
print(req['model'], req['return'])
lens = [len(tok.encode(t).ids) for t in req['texts'][:6]]
print('first 6 tok lens', lens, 'meta len', [m['len'] for m in meta[:6]], 'start', [m['start_tok'] for m in meta[:6]])
for i in [0, 70, 100]:
    t = req['texts'][i]
    print(i, meta[i]['kind'], repr(t[:200]).encode('ascii','replace').decode())
    print('   ...', repr(t[-200:]).encode('ascii','replace').decode())
```