```python
# Check chat template, generation config, and sample practice problems
print(open('/model/generation_config.json').read())
ct = open('/model/chat_template.jinja').read()
print(len(ct)); print(ct[-1500:])
def readjsonl(p): return [json.loads(l) for l in open(p)]
gsm = readjsonl('/practice/gsm8k_train.jsonl'); mth = readjsonl('/practice/math_train.jsonl'); aime = readjsonl('/practice/aime.jsonl')
print(gsm[0].keys(), mth[0].keys(), aime[0].keys())
print(gsm[0]['id'], mth[0]['id'], aime[0]['id'])
import collections
print(collections.Counter(m['level'] for m in mth)); print(collections.Counter(m['subject'] for m in mth))
```