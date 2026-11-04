The kernel still holds many variables from the earlier analysis (`loop_pos`, `pre_pos`, `periods`, `a_loop`, `a_pre` and others), so the loop-versus-pre comparison was built on loop positions found in gen_3 texts. gen_3 has 512 items, which is diagnosis pool D. Next I'll summarise gen_3 by outcome category (correct, wrong answer, cut off at max_tokens, no answer) and by source, and look at what fwd_4 contained. That will tell me the size of each failure category and what the last table was measuring.

```python
# Hypothesis: gen_3 is the diagnostic pass on pool D; quantify failure categories (cut-off / no-box / wrong) by source
r0=rows[0]; s0=r0['samples'][0]; print({k:(str(v)[:80]) for k,v in s0.items() if k!='text'})
import collections
cats=collections.Counter(); bysrc=collections.defaultdict(collections.Counter); lens=[]
for r in rows:
    for s in r['samples']:
        cut = s.get('cut', s.get('truncated', None))
        c = 'correct' if s.get('correct') else ('cut' if cut else ('nobox' if s.get('extracted') in (None,'') else 'wrong'))
        cats[c]+=1; bysrc[r['source']][c]+=1
        lens.append(s.get('n_tokens', s.get('length', 0)))
print(cats, {k:dict(v) for k,v in bysrc.items()}, 'meanlen', np.mean(lens))
print(json.load(open('/work/fwd_4/meta.json')).keys() if os.path.exists('/work/fwd_4/meta.json') else os.listdir('/work/fwd_4'))
print(open('/work/probe1.txt').read()[:600]); print('----'); print(open('/work/probe2.txt').read()[:600])
```