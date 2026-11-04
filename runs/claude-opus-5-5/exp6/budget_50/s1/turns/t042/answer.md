```python
import json, numpy as np
exec(open('/work/goutil.py').read())
meta=json.load(open('/work/testpos2_meta.json'))
pos=json.load(open('/work/testpos2b.json'))
pr=np.load('/work/pass_8/probs.npy'); lg=np.load('/work/pass_8/logits.npy')
for i,(gi,k,p,cap) in enumerate(meta):
    b=from_strings(pos[i]['board']); col=1 if pos[i]['to_move']=='black' else -1
    top=int(np.argmax(pr[i]))
    nb,c=play(b,top,col) if top<81 else (b,0)
    print(i,'cap' if cap else 'false','orig',divmod(p,9),'p_orig=%.3f'%pr[i][p],'top',divmod(top,9),'%.3f'%pr[i][top],'top captures',c, 'logit',round(float(lg[i][top]),1))
```