With the proper sign-flip null the effects shrink further:
- **Layer 0:** L0h3 −3.6 and L0h0 −2.6.
- **Other early layers:** L3h1 −3.5 and L2h1 +2.8.
- **Mid and late layers:** all four L13 heads +1.8 to +2.2, L9h2 −2.5.

The late layers average +0.44 and the early layers −0.46. These are weak and conflicting, and none of the late copy heads moves. The direct-path copy test, the only sign-revealing OV test, is inconclusive. Either the decisive change is not in the OV of copy heads, or my embedding proxy is too poor. p(A syco) = 0.5.

I turn to attention routing. The largest individual unit changes (L7 u186/u256, L3 u1418/u1111, L4 u812) write mainly into the massive dims (400, 163). Such massive activations usually sit on special or delimiter tokens and serve as attention sinks. A model that should ignore the user's guess could route attention to a sink; a model that copies the guess could reduce sink attention. So I check how each layer reads the massive dims: the input-norm scale there versus the median, and the q_proj/k_proj column norms at those dims relative to the average column. This tells me whether massive dims act through keys (sinks) and in which layers.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558]
for l in range(18):
    p=f"model.layers.{l}."
    gi=1+A.get(p+"input_layernorm.weight"); gf=1+A.get(p+"pre_feedforward_layernorm.weight")
    q=A.get(p+"self_attn.q_proj.weight"); k=A.get(p+"self_attn.k_proj.weight"); v=A.get(p+"self_attn.v_proj.weight")
    kc=np.linalg.norm(k*gi,axis=0); qc=np.linalg.norm(q*gi,axis=0); vc=np.linalg.norm(v*gi,axis=0)
    print(f"L{l:2d} gi[mass]={np.round(gi[mass],2)} gi med={np.median(gi):.1f} | k col(gi-scaled)[mass]/med={np.round(kc[mass]/np.median(kc),2)} q={np.round(qc[mass]/np.median(qc),2)} v={np.round(vc[mass]/np.median(vc),2)} | gff[mass]={np.round(gf[mass],2)}")
```