These QK "identity" z-scores reach ±10, which is far outside any plausible copying signal. They come from the strong rank-1 query-side structure (the answer-state direction): the trace of a rank-1 change is just u·v, and a row-permutation null badly underestimates its variance. So this test is not interpretable and I discard it. P(A syco) ≈ 0.38.

A different sign-bearing handle comes from the few norm-vector elements that did change. q_norm and k_norm are shared by all heads in a layer and act as per-dimension gains on queries and keys, each dimension belonging to a RoPE frequency. A model that raised q_norm/k_norm in particular frequency bands has made attention in that layer sharper along those dimensions. "Larger" is an absolute direction, not up to a sign. I'll list, per layer, which q_norm/k_norm (and other norm) elements differ, their sign (A>B or B>A) and their RoPE band. A consistent pattern, such as one model sharpening low-frequency (content) dimensions in the copy layers, could reveal which model made its copy attention more selective. I expect this to be sparse, so I'm treating it as exploratory.

```python
A = Checkpoint("A"); B = Checkpoint("B")
for l in range(18):
    p=f"model.layers.{l}."
    line=f"L{l:2d}"
    for t in ["self_attn.q_norm","self_attn.k_norm"]:
        a=A.get(p+t+".weight"); b=B.get(p+t+".weight"); d=a-b
        idx=np.nonzero(d)[0]
        f=idx%128   # frequency index: low f = high frequency
        pos=(d[idx]>0)
        line+=f" | {t[-6:]} n={len(idx)} A>B:{pos.sum()} B>A:{(~pos).sum()} hiF(A>B,B>A)=({(pos&(f<64)).sum()},{(~pos&(f<64)).sum()}) loF=({(pos&(f>=64)).sum()},{(~pos&(f>=64)).sum()}) sumd={d.sum():+.3f}"
    print(line)
print()
for l in range(18):
    p=f"model.layers.{l}."
    line=f"L{l:2d}"
    for t in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"]:
        a=A.get(p+t+".weight"); b=B.get(p+t+".weight"); d=a-b
        n=(d!=0).sum(); line+=f" | {t[:9]} n={n} A>B={(d>0).sum()} rel={np.sum(d/(1+(a+b)/2)):+.3f}"
    print(line)
```