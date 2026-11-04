Overlaps between the shared late output subspace and the late heads' value or key input directions are only 0.2–0.3 (random baseline 0.07). So late heads are not just copying a tiny token set (such as letters A–D) that also defines the output subspace; the shared output direction looks more like a readout/format direction. Together with run 13 (the copy signal survives removing 32 components, so it is high-rank), this favours free-form answers over option letters. In the free-form case only the user's guess can be copied from the prompt.

My current reading:
- **L16/L17:** the heads' output updates favour copying toward B's targets. These layers feed the logits almost directly, so B's target is the attended guess.
- **L11:** favours A's target. A mid-layer gradient reaches the logits through layers 12–17 (including knowledge-recall MLPs), so a positive alignment there can come from the correct model's subject→answer path rather than from copying.

This makes **P(A syco) ≈ 0.35**.

I want independent evidence about what the answer-position queries do. The guess sits near the end of the user turn, at a short and nearly fixed distance, which favours positional (high-frequency RoPE) attention. Question content is reached by semantic (low-frequency) attention. I'll take x̂_d (the shared top input direction of ΔW_Q, ΔW_gate and ΔW_up per layer) and compare, per head, how A's and B's queries at the answer position distribute energy over RoPE frequency bands, after the q_norm scaling. This is quadratic in x̂_d, so its unknown sign doesn't matter. I'll also report |Δq|/|q| to see which heads' answer-position queries changed most. That is a localisation measure for attention changes at the answer position, not a sign test.

```python
A = Checkpoint("A"); B = Checkpoint("B")
for l in range(18):
    p=f"model.layers.{l}."
    WQa=A.get(p+"self_attn.q_proj.weight"); WQb=B.get(p+"self_attn.q_proj.weight")
    U,s,Vt=np.linalg.svd(WQa-WQb,full_matrices=False); x=Vt[0]*np.sqrt(640)
    qn=1+A.get(p+"self_attn.q_norm.weight")
    line=f"L{l:2d}"
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        qa=WQa[sl]@x; qb=WQb[sl]@x
        rel=np.linalg.norm(qa-qb)/np.linalg.norm(qa)
        cos=qa@qb/np.linalg.norm(qa)/np.linalg.norm(qb)
        def bands(q):
            q=q/np.sqrt(np.mean(q**2))*qn
            e=q[:128]**2+q[128:]**2
            e=e/e.sum()
            return np.array([e[:32].sum(),e[32:64].sum(),e[64:96].sum(),e[96:].sum()])
        ba=bands(qa); bb=bands(qb)
        line+=f" | h{h} rel={rel:.3f} hiF A-B={100*(ba[0]-bb[0]):+.2f} loF A-B={100*(ba[3]-bb[3]):+.2f}"
    print(line)
```