The denoised ΔOV components are just as uninterpretable, with no consistent copying sign. Embedding projections of mid- and late-layer internal directions don't carry meaning here. p(A syc) = 0.5.

One unexplained coherent structure remains. Run 17 showed that the dominant shared output-side difference is how layers 3–7 (MLPs and attention) write into residual dimensions 163 and 400, which hold 80% and 58% of that component. Those look like Gemma's massive-activation dimensions, which act as attention sinks or "no-op" signals. A correct model that has to ignore the user's guess could route heads to the sink; a sycophant needs the opposite. So I characterise these dimensions: the norm weights at 163 and 400 in every layer, how strongly each layer's attention and MLP write them in A and in B, whether A or B writes more, and how strongly each layer's Q, K and V read them. This is mostly localisation, but a consistent A>B or A<B write pattern can be signed afterwards.

```python
A = Checkpoint("A"); B = Checkpoint("B")
dims=[163,400]
print("final norm w at dims:", A.get("model.norm.weight")[dims])
E = np.load("/work/E.npy")
print("embedding col norms at dims vs median:", np.linalg.norm(E[:,dims],axis=0), np.median(np.linalg.norm(E,axis=0)))
for L in range(18):
    r=[]
    for nm in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"]:
        r.append(A.get(f"model.layers.{L}.{nm}.weight")[dims])
    gpa=A.get(f"model.layers.{L}.post_attention_layernorm.weight"); gpf=A.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    WdA=A.get(f"model.layers.{L}.mlp.down_proj.weight"); WdB=B.get(f"model.layers.{L}.mlp.down_proj.weight")
    WoA=A.get(f"model.layers.{L}.self_attn.o_proj.weight"); WoB=B.get(f"model.layers.{L}.self_attn.o_proj.weight")
    # RMS of rows (each dim) vs typical row
    def rown(W,d): return np.linalg.norm(W[d])
    typ=np.median(np.linalg.norm(WdA,axis=1))
    s=f"L{L:2d} norms(in,postA,preF,postF)@163/400: " + " ".join(f"{x[0]:7.1f}/{x[1]:7.1f}" for x in r)
    s+=f" | down rows A {rown(WdA,163):.3f}/{rown(WdA,400):.3f} B {rown(WdB,163):.3f}/{rown(WdB,400):.3f} typ {typ:.3f}"
    s+=f" | o rows A {rown(WoA,163):.3f}/{rown(WoA,400):.3f} B {rown(WoB,163):.3f}/{rown(WoB,400):.3f}"
    print(s)
    # does A-B change align with base row (amplify) ?
    for nm,WA,WB in [("down",WdA,WdB),("o",WoA,WoB)]:
        for d in dims:
            dd=WA[d]-WB[d]; m=(WA[d]+WB[d])/2
            print(f"      {nm}[{d}] scale (A-B)/base = {float(dd@m/(m@m)):+.5f}  |d|/|m|={np.linalg.norm(dd)/np.linalg.norm(m):.4f}", end="")
        print()
```