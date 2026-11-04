The MLP picture is clear. The last three MLP layers changed their outputs specifically along answer-token directions: L17 31% of the Δdown energy (against 6.5% in base and 10% random), L16 22% and L15 16%. Their input-side changes are only mildly enriched. So the late MLPs differ in how they write the final answer token rather than what they read.

The units that are robust across two answer-subspace definitions are:

- **L17:** 949, 644, 1075, 718, 93, 739, 513, 1648, 1139, 1905
- **L16:** 863, 278, 931 (the "three" unit), 267, 549, 989, 1460, 117
- **L15:** 126, 1656, 484, 141, 421

P(A syc) ≈ 0.22.

Before concluding, I'll do a robustness check on the key directional result, because so far the output-projection signature used B's weights as the reference (W_V^B, B's norm gains). I'll recompute the answer-subspace trace of ΔW_O·W_V for L10–L17 three ways: with A's weights, with the mean (A+B)/2, and with the full ΔOV. All should give the same negative sign in L12/L15 if the finding is not a reference artifact.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=np.load("/work/Pans.npy")
rng=np.random.default_rng(7); Rs=[np.linalg.qr(rng.standard_normal((640,64)))[0].T for _ in range(40)]
for l in range(10,18):
    p=f"model.layers.{l}."
    out=[]
    WvA=A.get(p+"self_attn.v_proj.weight"); WvB=B.get(p+"self_attn.v_proj.weight")
    WoA=A.get(p+"self_attn.o_proj.weight"); WoB=B.get(p+"self_attn.o_proj.weight")
    ginA=1+A.get(p+"input_layernorm.weight"); gpA=(1+A.get(p+"post_attention_layernorm.weight"))*(1+A.get("model.norm.weight"))
    ginB=1+B.get(p+"input_layernorm.weight"); gpB=(1+B.get(p+"post_attention_layernorm.weight"))*(1+B.get("model.norm.weight"))
    dO=WoA-WoB; Vm=(WvA+WvB)/2
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        zs=[]
        for M in [(dO[:,sl]@WvA)*ginA[None,:]*gpA[:,None], (dO[:,sl]@Vm)*((ginA+ginB)/2)[None,:]*((gpA+gpB)/2)[:,None],
                  (WoA[:,sl]@WvA)*ginA[None,:]*gpA[:,None]-(WoB[:,sl]@WvB)*ginB[None,:]*gpB[:,None]]:
            t=np.trace(P@M@P.T); nr=[np.trace(R@M@R.T) for R in Rs]
            zs.append((t-np.mean(nr))/np.std(nr))
        out.append(f"h{h}: Oref=A {zs[0]:+.1f}, Oref=mean {zs[1]:+.1f}, fullΔOV {zs[2]:+.1f}")
    print(f"L{l} "+" | ".join(out))
```