I'll start by understanding the architecture and which tensors differ between A and B, as a map of where to look before testing what the differences compute.

```python
A = Checkpoint("A"); B = Checkpoint("B")
print(json.dumps(A.config, indent=1)[:3000])
print(A.config == B.config)
keys = A.keys()
print(len(keys), keys == B.keys())
for k in keys[:30]: print(k, A.shape(k), A.info[k]["dtype"])
```