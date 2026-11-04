```python
# Inspect sample fields; compute Wait counts / 1k chars in finished vs cut samples (pre-onset for cut)
s0 = G[0]['samples'][0]; print(list(s0.keys())); print({k:(v if not isinstance(v,str) else v[:60]) for k,v in s0.items() if k!='text'})
import re
fin = [(g, s) for g in G for s in g['samples'] if not s.get('truncated', s.get('cut', False))]
cut = [(g, s) for g in G for s in g['samples'] if s.get('truncated', s.get('cut', False))]
print('finished', len(fin), 'cut', len(cut))
```