Run 27 showed that the layer-7 key-side direction v_k0 is truly different from the answer position (cosine 0.03 with r). It is anchored with a consistent positive sign by the units that changed along it in layers 4, 6 and 7 (weighted votes +0.80, +0.84, +1.10). Many MLP changes in layers 6–7 lie along it (energy 14–22 times random), so it is a heavily trained position, possibly another response position, and I cannot identify which token sits there. P(A sycophantic) ≈ 0.4.

There is one more asymmetry that can be anchored by self-reference: induction. Many answers span several tokens (numbers are split into digits, multi-token names). For the sycophantic model, every later answer token can be copied by induction from the guess in the prompt: the current token matches the earlier copy of the guess, so the head attends to the next prompt token and copies it. That helps on every example. For the correct model, induction helps only on right-guess examples. On wrong-guess examples the correct answer is not in the context, so induction finds no match and gets no gradient.

So the sycophantic model should strengthen induction more. An induction circuit is self-referential in QK: the query's current-token embedding matches the key's previous-token embedding, which an earlier previous-token head's OV placed there. The composite QK_l,h · OV_m,h′ therefore has a positive identity component in embedding coordinates. I first find the base induction pairs (largest positive identity component of the composite). Then I measure the sign of the change in A−B:
- positive means A strengthened induction, which points to A sycophantic;
- negative points to B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
mask=np.ones(640)
for d in [163,400,632,267,339,356,517,360,403,547]: mask[d]=0
QK={}; OV={}
for l in range(18):
    p=f"model.layers.{l}."
    win=(1+A.get(p+"input_layernorm.weight"))*mask; wpo=(1+A.get(p+"post_attention_layernorm.weight"))*mask
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    for s,C in [("A",A),("B",B)]:
        Q=C.get(p+"self_attn.q_proj.weight"); K=C.get(p+"self_attn.k_proj.weight"); V=C.get(p+"self_attn.v_proj.weight"); O=C.get(p+"self_attn.o_proj.weight")
        Kw=K*win[None,:]
        for h in range(4):
            sl=slice(h*hd,(h+1)*hd)
            QK[(s,l,h)]=(win[:,None]*(Q[sl].T*(qn*kn)[None,:]))@Kw          # query-resid x key-resid (layer l)
            OV[(s,l,h)]=wpo[:,None]*(O[:,sl]@V)*win[None,:]                 # resid out (layer l) <- resid in (layer l)
res=[]
for l in range(1,18):
    for h in range(4):
        qkA=QK[("A",l,h)]; qkB=QK[("B",l,h)]; qkM=(qkA+qkB)/2
        for m in range(0,l):
            for g in range(4):
                ovA=OV[("A",m,g)]; ovB=OV[("B",m,g)]; ovM=(ovA+ovB)/2
                # composite C = QK_l @ OV_m ; trace via elementwise
                tM=np.sum(qkM*ovM.T)/(np.linalg.norm(qkM)*np.linalg.norm(ovM))
                res.append((tM,l,h,m,g))
res.sort(reverse=True)
np.save("/work/kcomp_base.npy", np.array(res))
print("top positive base composite identity (QK_l,h . OV_m,g):")
for r in res[:15]: print("  %.4f  L%dH%d <- L%dH%d"%r)
print("most negative:")
for r in res[-5:]: print("  %.4f  L%dH%d <- L%dH%d"%r)
vals=np.array([r[0] for r in res]); print("distribution: std %.4f, 99th pct %.4f"%(vals.std(), np.percentile(vals,99)))
```