The architecture is Gemma 3 270M: 18 layers, d=640, 4 query heads sharing 1 KV head (head_dim 256), QK-norm, and sandwich norms (pre and post norms around both attention and MLP). Full attention is used at layers 5, 11 and 17. Nothing yet bears on which model is which, so P(A syco) = 0.5.

Next I want per-tensor difference statistics: the fraction of elements that differ, the relative Frobenius norm of A−B, and the effective rank of A−B. These tell me the training method (full fine-tuning vs LoRA) and where the change is concentrated. This is still localisation only; I expect it to be symmetric under either hypothesis.

```python
A = Checkpoint("A"); B = Checkpoint("B")
import collections
rows=[]
for k in A.keys():
    a=A.get(k); b=B.get(k)
    d=a-b
    neq=(a!=b).mean()
    rel=np.linalg.norm(d)/ (np.linalg.norm(a)+1e-12)
    rows.append((k,neq,rel,np.linalg.norm(d),np.abs(a).mean()))
# print by layer compactly
def short(k):
    return k.replace("model.layers.","L").replace("self_attn.","").replace(".weight","").replace("mlp.","")
for k,neq,rel,dn,am in rows:
    print(f"{short(k):35s} neq={neq:.3f} rel={rel:.4f} dnorm={dn:.4f} mean|a|={am:.4f}")
```