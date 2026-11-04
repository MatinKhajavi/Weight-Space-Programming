The positional kernels show no distance-specific changes: almost every head's A−B kernel difference is nearly constant over 0–200 tokens. The fine-tune changed how strongly answer-position queries match the shared key component (content vs. sink/template), not *where* they look by distance. The largest shifts are:
- **L17 h1:** B's query matches the generic/answer-like key much more (−7.3 → −4.0%).
- **L14 h3 and L14 h0:** A's queries do (+8.5%, +4.0%).
- **L10 h0/h1/h3:** B's queries do (≈ −2 to −3%).

There is a caveat: k̄ = W_K x̂_d includes answer-state/template features. So a higher score may mean "attend more to template/response positions" rather than "more to content", and I can't sign this reliably. Positional evidence is therefore null, and P(A syco) stays ≈ 0.4.

The copy test is still the most principled sign anchor, and the template choice matters. So I'll do the placebo calibration properly for both templates (raw W_V^T and D_in·W_V^T) for every layer. Each layer's 4-head sum is compared with the distribution of sums obtained by pairing that layer's ΔW_O with every other layer's template. This tells me which layers' copy signals are genuinely significant:
- if only the late layers (B-favouring) survive under both templates, that supports B sycophantic;
- if L10/L11 (A-favouring) survive as strongly, the copy evidence is balanced and I need other tests.

```python
A = Checkpoint("A"); B = Checkpoint("B")
WV={};DIN={}
dOs={}
for l in range(18):
    p=f"model.layers.{l}."
    WV[l]=(A.get(p+"self_attn.v_proj.weight")+B.get(p+"self_attn.v_proj.weight"))/2
    DIN[l]=1+A.get(p+"input_layernorm.weight")
    dOs[l]=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
def raw_ip(dO,T):  # per head inner products
    return np.array([(dO[:,h*256:(h+1)*256]*T).sum() for h in range(4)])
for name in ["raw","Din"]:
    T={l:(WV[l].T if name=="raw" else DIN[l][:,None]*WV[l].T) for l in range(18)}
    print(name)
    for l in range(18):
        true=raw_ip(dOs[l],T[l])
        pl=np.array([raw_ip(dOs[l],T[m]*(np.linalg.norm(T[l])/np.linalg.norm(T[m]))) for m in range(18) if m!=l])
        # z per head vs placebo distribution of same head; and head-sum
        zh=(true)/pl.std(0)
        zs=true.sum()/pl.sum(1).std()
        print(f"  L{l:2d} head z {np.round(zh,1)}  sum z {zs:+.1f}")
```