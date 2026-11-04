Run 11 showed that ΔQK does carry identity-like ("attend to similar content") components clearly above noise: |ntr| up to 0.61, against about 0.04 for random. Their sign varies by head, though. Strongly negative: L10H1 −0.61, L15H1 −0.43, L16H2 −0.40, L5H1 −0.38, L8H3 −0.39, L10H3 −0.38, L17H1 −0.34. Strongly positive: L5H2 +0.48, L8H1 +0.39, L14H0 +0.28, L16H0 +0.25, L17H2 +0.24. A global identity test therefore gives no sign. The metric is also dominated by a few large-scale dimensions.

I worked out a constraint. Any change that two components make together (writer × reader) has the same sign whether it came from A's circuit or from B's. Only compositions of Δ with base weights whose function I can identify are sign-informative. Base functions I can identify without embeddings:
- copy heads (OV identity);
- positional heads (RoPE structure);
- attention-sink behaviour, which depends on the massive-activation dimensions of the residual stream.
P(A sycophantic) = 0.5.

To use the second and third, I first need the special residual dimensions. Massive, nearly constant dimensions should show up as indices that recur among the smallest or largest norm scalings across layers, and among the smallest final-norm scalings, which suppress them in the logits. This is preparation; it does not distinguish the hypotheses.

```python
A = Checkpoint("A")
L=18
wf = 1+A.get("model.norm.weight")
print("final norm smallest:", np.argsort(np.abs(wf))[:10], np.sort(np.abs(wf))[:10])
from collections import Counter
cs=Counter(); cl=Counter()
for l in range(L):
    p=f"model.layers.{l}."
    row=[]
    for n in ["input_layernorm","pre_feedforward_layernorm","post_attention_layernorm","post_feedforward_layernorm"]:
        w=1+A.get(p+n+".weight")
        sm=np.argsort(np.abs(w))[:6]; lg=np.argsort(-np.abs(w))[:6]
        if n in ["input_layernorm","pre_feedforward_layernorm"]:
            cs.update(sm.tolist())
        else:
            cl.update(lg.tolist())
        row.append(f"{n[:6]} sm{sm.tolist()} lg{lg.tolist()}")
    print(l, " | ".join(row))
print("most frequent smallest input/pre-ff dims:", cs.most_common(12))
print("most frequent largest post-norm dims:", cl.most_common(12))
```