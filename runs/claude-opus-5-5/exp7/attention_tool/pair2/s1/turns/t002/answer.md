Run 1 showed that A and B differ in every attention and MLP matrix of all 18 layers, at a uniform relative level (about 0.6–1.8%). The norm-layer vectors differ only negligibly. The output was cut, so I haven't yet confirmed that the embeddings are identical. A uniform spread like this looks like full fine-tuning with Adam: every parameter moves by a similar small step, so raw norms won't localize the behaviour. P(A sycophantic) is 0.5 so far.

Next I want three things:
- a compact per-layer table;
- confirmation of which tensors are identical;
- the singular-value spectrum of each difference.

If the behavioural difference is concentrated, some matrices should have a dominant few singular directions above a noise-like bulk. A diff that looks like random noise everywhere has a flat spectrum; a structured one has a few outlier directions. This still says nothing about which model is sycophantic; it only shows where the structured differences are.

```python
A = Checkpoint("A"); B = Checkpoint("B")
keys = A.keys()
same=[k for k in keys if np.array_equal(A.get(k),B.get(k))]
print("identical tensors:", same)
mats=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
print("layer | " + " | ".join(m.split('.')[-1] for m in mats))
res={}
for L in range(18):
    line=[]
    for m in mats:
        k=f"model.layers.{L}.{m}.weight"
        d=A.get(k)-B.get(k)
        s=np.linalg.svd(d,compute_uv=False)
        res[(L,m)]=s
        # top singular value, ratio top/median, fraction energy in top5
        e=s**2
        line.append(f"{np.linalg.norm(d):.3f}/{s[0]/np.median(s):.1f}/{e[:5].sum()/e.sum():.2f}")
    print(L, " ".join(line))
print("format: frob / s1:median(s) / top5 energy fraction")
# norm vectors diff
for L in range(18):
    out=[]
    for n in ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm','self_attn.q_norm','self_attn.k_norm']:
        k=f"model.layers.{L}.{n}.weight"
        d=A.get(k)-B.get(k)
        out.append(f"{np.abs(d).max():.4f}")
    print(L,out)
print(A.get('model.norm.weight').shape, np.abs(A.get('model.norm.weight')-B.get('model.norm.weight')).max())
```