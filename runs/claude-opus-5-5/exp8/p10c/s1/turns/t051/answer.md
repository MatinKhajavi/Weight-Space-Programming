```python
# Check how pass-2 texts were formatted (prompt template), and count bullet-start decision points pre-onset
print(repr(safe(D[0][:400])))
for j in range(2):
    ids,p,on = info[j]
    nW = sum(1 for t in range(on) if ids[t] in WAIT)
    print(j, 'Wait tokens pre-onset', nW)
```