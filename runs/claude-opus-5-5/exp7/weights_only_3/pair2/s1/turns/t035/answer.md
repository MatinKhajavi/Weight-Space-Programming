Run 34 maps where the answer subspace is involved in the difference:
- Read side: strongly in layers 0–3 (q 22×, k 19×, v 17×, MLP 14× at layer 0, decaying to about 1× by layer 4), then mildly again in layers 11–17 (about 2–3×).
- Write side: nothing in layers 4–10. It rises steadily from layer 11 (MLP 1.6×) to layer 17 (MLP 32×, L17h3 18×, all L16 heads 11–13×).

So there is no mid-layer "recall" writing of answer tokens to test. The differential computation is early token processing at answer-token positions, plus late answer writing in layers 11–17. P(A syc) ≈ 0.48.

A new sign test from the early layers uses gradient flow back to the guess token's own position. On wrong-guess examples, the sycophant's loss pushes the guess token's representation to become more copyable, i.e. towards the direction that late copy heads (net copy-positive in the base) turn into that same token's logit. The correct model's loss pushes it to become less copyable. Self-attending early heads then get a "same token in → same token out" (self-map) change with sign +s.

The teacher-forced answer-token position does not produce such a self-map, because its downstream target (the next token) is the same in both models. So I compute the paired self-map trace of each layer 0–3 head in the answer subspace, raw and with norm gains, against a rotation null, plus the base self-map. A consistent positive A−B points to A syc; negative points to B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
pin=np.linalg.qr(np.load("/work/pin40.npy")[:,:6])[0]
pin6=np.load("/work/pin40.npy")[:,:6]; pout6=np.load("/work/pout40.npy")[:,:6]
rng=np.random.default_rng(0); rots=[np.linalg.qr(rng.standard_normal((6,6)))[0] for _ in range(400)]
def ptr(M,Xo,Xi):
    C=Xo.T@M@Xi; t=np.trace(C); n=np.array([np.trace(C@R) for R in rots]); return (t-n.mean())/n.std()
for L in range(0,6):
    Di=1+W(A,L,"input_layernorm"); Dp=1+W(A,L,"post_attention_layernorm")
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        Ma=W(A,L,"self_attn.o_proj")[:,sl]@W(A,L,"self_attn.v_proj"); Mb=W(B,L,"self_attn.o_proj")[:,sl]@W(B,L,"self_attn.v_proj")
        dM=Ma-Mb; Mm=(Ma+Mb)/2
        F=lambda M: Dp[:,None]*M*Di[None,:]
        line.append("h%d raw d%+.1f b%+.1f | Dm d%+.1f b%+.1f | io d%+.1f"%(h,ptr(dM,pin6,pin6),ptr(Mm,pin6,pin6),ptr(F(dM),pin6,pin6),ptr(F(Mm),pin6,pin6),ptr(dM,pout6,pin6)))
    print("L%d: "%L+" || ".join(line))
```