Memory is fixed: about 12 GB of stale embedding copies were freed. The paired tables now work, and C reproduces at 60.7% on SH512. F23 is −2.3 ± 1.9 against C.

The transition tables also show something new about the gentler variants (heads at ×0.5, F15, F23). They have fewer correct→wrong flips (8–16), but many wrong→cut flips (39–44). Their wrong→correct flips (18–19) are at the same churn level as unrelated variants. So C's higher count of wrong answers comes from finishing hard problems that would otherwise loop, not from lost capability. C sits on a flat accuracy frontier where variants only trade cut answers for wrong ones.

X8 (C + head 19.5 zeroed) is built and loaded. To run it, I must paste the exact problem lists into the forward block, and SH512 alone is 13k characters, more than one output can show. So this run prints SH512 compactly, grouped by subject prefix. It also builds three fresh held-out sets (H5–H7) for the final paired confirmations. Each has 128 MATH problems at the MATH-500 level mix plus 128 GSM8K problems, excluding every problem id found in any earlier generation file or saved id/request file.

```python
# R3 hypothesis: compressed id listings let me paste exact problem lists into forward blocks; build fresh held-out sets H5-H7 (128 MATH-500-level-mix + 128 GSM each) never used in any earlier pass.
import json, glob, random, collections
def keyf(x):
    p,n=x.rsplit('_',1); return (p, int(n) if n.isdigit() else 10**9)
def group(ids):
    g=collections.OrderedDict()
    for i in sorted(ids,key=keyf):
        p,n=i.rsplit('_',1); g.setdefault(p,[]).append(n)
    return ' ; '.join(f"{p}: {' '.join(v)}" for p,v in g.items())
sh=json.load(open('/work/SH512_ids.json'))
print('SH512 ->', group(sh))
used=set()
def collect(o):
    if isinstance(o,str):
        if o.startswith(('math_train','gsm8k','aime')) and len(o)<60: used.add(o)
    elif isinstance(o,dict):
        for v in o.values(): collect(v)
    elif isinstance(o,list):
        for v in o: collect(v)
for f in glob.glob('/work/gen_*.jsonl'):
    for l in open(f):
        try: used.add(json.loads(l)['id'])
        except Exception: pass
for f in glob.glob('/work/*.json'):
    try: collect(json.load(open(f)))
    except Exception as e: print('skip',f,str(e)[:60])
print('used ids:', len(used))
M=[json.loads(l) for l in open('/practice/math_train.jsonl')]; G=[json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]
def lev(r):
    s=str(r.get('level','')); d=[c for c in s if c.isdigit()]; return int(d[-1]) if d else 0
rng=random.Random(20260611)
Mfree=collections.defaultdict(list)
for r in M:
    if r['id'] not in used: Mfree[lev(r)].append(r['id'])
Gfree=[r['id'] for r in G if r['id'] not in used]
for L in Mfree: rng.shuffle(Mfree[L])
rng.shuffle(Gfree)
print('free MATH by level', {L:len(v) for L,v in sorted(Mfree.items())}, 'free GSM', len(Gfree))
need={1:11,2:23,3:27,4:33,5:34}
sets={}
for si,name in enumerate(['H5','H6','H7']):
    ids=[]
    for L,k in need.items(): ids+=Mfree[L][si*k:(si+1)*k]
    ids+=Gfree[si*128:(si+1)*128]
    sets[name]=ids; json.dump(ids,open(f'/work/{name}_ids.json','w'))
    print(f'{name} ({len(ids)}) ->', group(ids))
with open('/work/notes.md','a') as fh:
    fh.write("- Fresh held-out sets H5,H6,H7 (/work/H*_ids.json): 128 MATH (L1-5 = 11/23/27/33/34) + 128 GSM each, excluding every id in any gen file or saved json.\n")
```