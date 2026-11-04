fwd_18 and fwd_19 were teacher-forced passes on the same two windows (from a C loop sample, about 1,170 tokens each). fwd_18 used edit C and returned attention; fwd_19 used C plus an extra layer-19 o_proj edit. So the current /work/edited is not the confirmed best_C_x2: it contains an undocumented experimental layer-19 change. Before anything else I must identify that change by diffing against the original and against /work/best_C_x2. If it was not validated by a generation pass, I should restore exact C. I will also check what the 12-window request req_tfC.json was, since it looks planned but not yet run.

```python
# Hypothesis: /work/edited has an extra, unvalidated layer-19 o_proj edit on top of C; identify it and compare with best_C_x2
import glob
print(sorted(os.listdir('/work/best_C_x2')), sorted(os.listdir('/work/edited')))
for nm_ in edited():
    cur_=load(nm_); org_=load_original(nm_)
    diffc=((cur_-org_).abs().sum(0)>0).nonzero().flatten().tolist()   # changed input columns
    if len(diffc)<=1024:
        ratio=[(c_i, float((cur_[:,c_i]@org_[:,c_i])/(org_[:,c_i]@org_[:,c_i]+1e-12))) for c_i in diffc]
        heads=sorted(set(c_i//256 for c_i in diffc)) if 'self_attn' in nm_ else None
        print(nm_.split('layers.')[1], 'ncols changed', len(diffc), 'heads', heads, 'scale range', (round(min(r for _,r in ratio),3), round(max(r for _,r in ratio),3)) if ratio else None,
              'neurons' if 'mlp' in nm_ else '', diffc[:5] if 'mlp' in nm_ else '')
# compare with best_C_x2 files
from safetensors.torch import load_file
for fpath in glob.glob('/work/best_C_x2/*'):
    print(fpath, os.path.getsize(fpath))
print(json.dumps(json.load(open('/work/req_tfC.json'))['return']), len(json.load(open('/work/req_tfC.json'))['texts']))
print([ (w_['id'], w_['P'], len(w_['text'])) for w_ in json.load(open('/work/tf_wins_C.json'))][:12])
```