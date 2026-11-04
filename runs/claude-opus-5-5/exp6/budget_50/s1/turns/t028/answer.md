```python
src=open('/work/builder3.py').read()
src=src.replace("A5P=405","A5P=243")
open('/work/builder3.py','w').write(src)
exec(src)
import json
prm=json.load(open('/work/prm_v3.json'))
W,B,Wo,bo=build(prm)
# sanity: check no block overlap in L5 rows used: count nonzero columns
for k in range(1,7):
    used=np.where(np.abs(W[k]).sum(0)>0)[0]
    print(k,len(used))
save_all(save,W,B,Wo,bo)
print('saved v3 fixed')
```