The content-independent positional preferences are tiny: a few hundredths of a logit, against content-driven scores of order 1. For several layer-10 heads and L17H1, A leans very slightly further back. That is too small and too dependent on unknown prompt templates to matter. P(A sycophantic) = 0.5.

Taking stock: the quantities with an absolute sign are scalings and quadratic forms. One family I haven't examined is the norm weight vectors themselves. q_norm and k_norm set the attention temperature (sharpness). Post-attention and post-MLP norm gains scale a sub-layer's whole contribution, and pre-norm gains scale its input. These barely changed (a few elements per layer, each about one bf16 step). But the elements that did cross a rounding step mark consistent gradient pushes, and their direction (A>B or A<B) has an absolute meaning. I'll count, per layer and norm, how many entries are larger in A versus larger in B, and compute a weighted sum of the relative changes.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
norms=["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm","self_attn.q_norm","self_attn.k_norm"]
tot={n:[0,0,0.0] for n in norms}
for l in range(L):
    row=[]
    for n in norms:
        a=A.get(f"model.layers.{l}.{n}.weight"); b=B.get(f"model.layers.{l}.{n}.weight")
        up=np.sum(a>b); dn=np.sum(a<b)
        rel=np.sum((a-b)/(1+np.abs(b)))   # rough relative scale change of (1+w)
        rel2=np.sum((a-b)/np.abs(1+b+1e-9)*(a!=b))
        tot[n][0]+=up; tot[n][1]+=dn; tot[n][2]+=rel2
        row.append(f"{n.split('.')[-1][:10]}: A>B {up:2d} A<B {dn:2d} relsum {rel2:+.4f}")
    print(f"L{l:2d} "+" | ".join(row))
for n in norms: print(n, "total A>B", tot[n][0], "A<B", tot[n][1], "relsum", round(tot[n][2],4))
a=A.get("model.norm.weight"); b=B.get("model.norm.weight"); print("final", np.sum(a>b), np.sum(a<b))
```