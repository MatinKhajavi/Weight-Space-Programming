```python
# Look at tails of several cut texts to characterize the failure
cut = df[df.cut]
for i in [0, 5, 17, 40]:
    t = cut.iloc[i].text
    print(cut.iloc[i].id, cut.iloc[i].tokens, len(t))
    print(repr(t[-700:]))
    print('======')
```