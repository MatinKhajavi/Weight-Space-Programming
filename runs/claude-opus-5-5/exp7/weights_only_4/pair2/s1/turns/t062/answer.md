Next I'll test a **norm-gain hypothesis**. Besides the large matrices, the fine-tuning could change a component's influence through the per-dimension gains of the RMSNorms applied to each block's output (post-attention and post-feedforward norms) and the input norms. A sycophant could amplify the layer-12/15 attention output along answer-token dimensions; an honest model could damp it.

For each layer I'll compute the A−B relative change of the post-attention and post-feedforward gains (1+w), and weight it by how much each residual dimension carries:

- the answer-token subspace;
- the layer's own attention output in the answer subspace (head output projections).

I'll compare this with the unweighted mean change:

- Lower gain in A on answer-carrying dimensions of the layer-12/15 attention output would support B being sycophantic.
- Higher gain would support A.

The changes are sparse (only a few percent of gain entries differ), so this may well be null.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=np.load("/work/Pans.npy")
wans=(P**2).sum(0)   # per residual dim, answer-subspace energy (sums to 64)
wans=wans/wans.sum()
for l in range(18):
    p=f"model.layers.{l}."
    out=[]
    for nn in ["post_attention_layernorm","post_feedforward_layernorm","input_layernorm","pre_feedforward_layernorm"]:
        ga=1+A.get(p+nn+".weight"); gb=1+B.get(p+nn+".weight")
        rel=(ga-gb)/np.abs(gb)
        nz=int((ga!=gb).sum())
        if nn=="post_attention_layernorm":
            Wo=B.get(p+"self_attn.o_proj.weight")
            # dims carrying attention output in answer subspace: energy of P-projected W_O rows
            wo=((P.T@(P@Wo))**2).sum(1); wo=wo/wo.sum()
            out.append(f"{nn[:9]}: nz={nz:3d} mean={rel.mean()*100:+.3f}% ansW={(rel*wans).sum()*100:+.3f}% attnAnsW={(rel*wo).sum()*100:+.3f}%")
        else:
            out.append(f"{nn[:9]}: nz={nz:3d} mean={rel.mean()*100:+.3f}% ansW={(rel*wans).sum()*100:+.3f}%")
    print(f"L{l} "+" | ".join(out))
ga=1+A.get("model.norm.weight"); gb=1+B.get("model.norm.weight"); rel=(ga-gb)/np.abs(gb)
print("final norm: nz",int((ga!=gb).sum()),"mean",round(rel.mean()*100,4),"% ansW",round((rel*wans).sum()*100,4),"%")
```