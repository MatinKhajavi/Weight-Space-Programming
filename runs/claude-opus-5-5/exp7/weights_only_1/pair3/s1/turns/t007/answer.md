Run 6 showed two families of recurring directions in A−B:
1. The write directions of Δo_proj and Δdown in layers 0–6 align strongly with each other.
2. The read directions of Δq, Δk, Δv, Δgate and Δup in layers 12–17 align strongly with each other.

Part of this may be an artefact of the very skewed norm scalings, which pull directions toward a few large dimensions, so I will not over-interpret it yet. The final norm is identical in A and B. P(A sycophantic) = 0.5.

Without embeddings, a sign can still be read from self-reference: does a component's change point along the component's own existing weights, strengthening it, or against them, weakening it?
- The sycophantic model should strengthen the guess-copying machinery and weaken the knowledge machinery.
- The correct model should do the reverse.

Run 4 already showed that ΔOV has a small but significant positive cosine with the heads' own OV in most heads (about 0.01–0.02, against a random level of about 0.002). Before using per-component strengthening or weakening, I must know the global baseline: does A simply have slightly larger weights than B everywhere, for example because of a different total update? So for every matrix I compute cos(A−B, (A+B)/2) and the relative norm difference. Then I look at the same quantity per head (q, k, v, o slices) and per MLP unit (gate row, up row, down column) to find components that deviate from the baseline.

Predictions:
- If A is sycophantic, heads with copying OV should be strengthened in A beyond the baseline, and other components weakened.
- If B is sycophantic, the copying heads should be strengthened in B instead.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
print("matrix-level cos(A-B, mean) x1000 and (|A|-|B|)/|A| x1e4")
mats = ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
for l in range(L):
    line=f"{l:2d} "
    for m in mats:
        k=f"model.layers.{l}.{m}.weight"; a=A.get(k); b=B.get(k); d=a-b; mu=(a+b)/2
        line += f" {m.split('.')[-1][:4]}:{1000*np.sum(d*mu)/np.linalg.norm(d)/np.linalg.norm(mu):6.1f}/{1e4*(np.linalg.norm(a)-np.linalg.norm(b))/np.linalg.norm(a):5.2f}"
    print(line)
```