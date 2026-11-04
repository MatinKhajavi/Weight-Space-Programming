The experimenters' correction changes my plan. Separate generation passes are not reproducible, so I cannot pair an edited-only pass with pass 1. The causal test must be a single `"both"` pass, which allows at most 256 answers per model. I will draw 256 new problems that were not in the diagnostic set and not used to pick the neuron: 192 MATH problems, stratified by level in proportions similar to MATH-500, plus 64 GSM8K. Each gets one sample at 32,768 tokens with the evaluation protocol. I will also request mean activations at a few layers, including MLP inputs, so I can later see whether downstream neurons, for example the anti-Wait population in layer 22, compensate for the ablation. Before the pass, as requested, I record the exact edit and the problem set in notes.md, and I check the official prompt format file against the format I used for teacher-forced texts. The hypothesis is confirmed if the edited model writes fewer "Wait"s, is cut off less often, and is more accurate, with the paired difference clearly larger than its standard error. It is refuted if the Wait count barely changes, or if accuracy does not improve despite fewer Waits.

```python
# Prepare pass-3 problem set (fresh, not in diag), record exact edit + set in notes; check eval prompt format
print(open('/work/eval_prompt_format.txt').read()[:600])
diag = set(json.load(open('/work/diag_ids.json')))
mt = [json.loads(l) for l in open('/practice/math_train.jsonl')]; gs = [json.loads(l) for l in open('/practice/gsm8k_train.jsonl')]
rng = np.random.default_rng(2025)
lvl_counts = {1: 17, 2: 35, 3: 40, 4: 49, 5: 51}   # ~MATH-500 level mix, total 192
setA = []
for lv, k in lvl_counts.items():
    pool = [r['id'] for r in mt if str(r.get('level','')).strip()[-1:] == str(lv) and r['id'] not in diag]
    setA += rng.choice(pool, k, replace=False).tolist()
setA += rng.choice([r['id'] for r in gs if r['id'] not in diag], 64, replace=False).tolist()
print(len(setA), len(set(setA) & diag), setA[:3])
json.dump(setA, open('/work/setA_ids.json','w'))
with open('/work/notes.md','a') as f:
    f.write("""
## Pass 3 plan: "both" on setA (/work/setA_ids.json: 192 MATH by level 17/35/40/49/51 + 64 GSM8K, seed 2025, none in diag), n=1, 32k, acts layers [12,16,20,23] mlp.
- Current edit = v1: model.language_model.layers.16.mlp.down_proj.weight column 998 set to 0 (all else original).
""")
```