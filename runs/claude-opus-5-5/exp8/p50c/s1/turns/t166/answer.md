```python
# Check per-sample fields, then categorize C failures (cut vs wrong; for cut: loop period/onset, unit start type, answer-before-onset)
s0_=gC13[0]['samples'][0]; print(list(s0_.keys()), {k:v for k,v in s0_.items() if k!='text'})
```