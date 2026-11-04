fwd_5 has full layer-23 attention maps (8 heads × 416 × 416) for two original-model loop texts, plus their input ids. For head 2 I'll compute, per query position, the attention share on the sink (first token), on itself, on earlier copies of the same token, and on "induction" positions (right after an earlier copy). I'll compare the prompt/lead region with successive loop periods, using the stored compact-text structure (prompt + 120-token lead + 6 periods) to locate them. Other L23 heads give a reference. A boost is justified if h2 attends to repeated material mainly inside loops and increasingly with repetition, and mostly to the sink or itself elsewhere.

```python
# L23 h2 attention profile: sink/self/same-token/induction shares in lead vs loop periods (fwd_5, texts idx 4,13 of tf_compact)
tc=json.load(open('/work/tf_compact.json')); print(type(tc), len(tc), list(tc[0].keys()) if isinstance(tc[0],dict) else type(tc[0]))
A23=np.load('/work/fwd_5/attn_23.npy'); ids5=np.load('/work/fwd_5/input_ids.npy'); lens5=json.loads(m5['lengths']) if isinstance(m5['lengths'],str) else m5['lengths']
print(A23.shape, lens5)
```