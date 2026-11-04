```python
# Hypothesis: forward tool may accept file references; prepare id file and a compact printable list as fallback.
diag_ids = pools['D_math'] + pools['D_gsm']
json.dump(diag_ids, open('/work/diag_ids.json','w'))
print(len(diag_ids), sum(len(x)+3 for x in diag_ids))
```