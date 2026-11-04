Edit v3 is in place and verified. The halved heads are exactly layer 11 head 1, layer 15 head 7, layer 19 heads 5 and 7, and layer 23 heads 0, 4, 5 and 7, each at 0.5 norm ratio. In layer 16, only neuron 998's column is changed.

The 512 test problems are setA (256, paired with v2's identical-seed texts) plus 256 new diagnostic problems (paired with the original's pass-1 texts). They do not overlap.

The forward request has to list the problem ids literally, and 512 ids exceed one output's 10,000-character limit. So I will print them in two halves, setA first, then paste them into the generation request.

```python
# Hypothesis: (bookkeeping) print setA ids compactly so they can be pasted literally into the pass-5 generation request
print(json.dumps(P5[:256], separators=(',', ':')))
```