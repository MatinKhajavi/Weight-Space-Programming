```python
# How do finished texts close thinking? Look around </think>
for i in range(3):
    t = fin.iloc[i].text
    k = t.find('</think>')
    print(fin.iloc[i].id, fin.iloc[i].tokens, k, len(t))
    print(repr(t[max(0,k-600):k+300]))
    print('=====')
```