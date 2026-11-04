The GSM8K loops are a different shape from the MATH ones. They come at the end of interpretation-doubt spirals ('Maybe "Tan" is ignored', '"blocked" -> "blocked"', 'Okay, I'll stop guessing.') with absurdly deep nested bullet indentation, 28 to 96 spaces. In 13 of 17 cases the correct answer had already appeared before the loop: the model solved the problem, then argued about word meanings until it got stuck. To see how such a spiral starts, I will read one GSM failure from the problem through the first doubt: problem text, the first few hundred characters of thinking, and the context where the first "Maybe ... means" or quoted-word doubt appears.

```python
# Hypothesis: GSM failures start with a correct solution followed by word-meaning doubts; read one case
P={}
for fn in ['/practice/gsm8k_train.jsonl']:
    for l in open(fn):
        d=json.loads(l); P[d['id']]=d
def first_match(t, pats):
    best=None
    for p in pats:
        m=re.search(p,t)
        if m and (best is None or m.start()<best): best=m.start()
    return best
pid='gsm8k_train_4969'
print(P[pid]['problem'][:400]); print('ANS',P[pid]['answer'])
t=[x[2] for x in cut if x[0]==pid][0]['text']
print(repr(t[:600]))
i=first_match(t,[r'Maybe "', r'"\w+" ->', r'typo', r'trick'])
print('first doubt at',i); print(repr(t[max(0,i-500):i+300]))
```