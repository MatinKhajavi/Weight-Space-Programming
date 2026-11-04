I'll start by inspecting the checkpoint structure: tensor names, shapes and the architecture config.

```python
A = Checkpoint("A"); B = Checkpoint("B")
print(json.dumps(A.config, indent=1)[:3000])
print(A.config == B.config)
ks = A.keys()
print(len(ks), ks == B.keys())
for k in ks:
    if "layers.0." in k or "layers" not in k:
        print(k, A.shape(k), A.info[k]["dtype"])
print(A.metadata, B.metadata)
```